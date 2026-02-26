# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import torch
from typing import Dict, Optional
from torchinfo import summary

from model import build_model


def is_improvement(value: float, best: float, mode: str, epsilon: float) -> bool:
    """
    Check if a metric value represents an improvement over the best value.
    """
    if mode == "min":
        return (best - value) > epsilon
    else:
        return (value - best) > epsilon


class TrainCheckpoint:
    """
    TrainCheckpoint saves the model checkpoint when there is an improvement in the monitored metric.

    Args:
    - filepath: path to save the checkpoint
    - mode: "min" or "max", whether the monitored metric should be minimized or maximized
    - epsilon: minimum improvement to consider as an actual improvement (default: 0.0)
    """

    def __init__(self, filepath, mode: str = "min", epsilon: float = 0.0) -> None:
        assert mode in {"min", "max"}

        self.filepath = filepath
        self.mode = mode
        self.epsilon = epsilon
        self.reset()

    def reset(self) -> None:
        self.best_value = float("inf") if self.mode == "min" else -float("inf")

    def step(self, value: float, checkpoint_dict: Dict) -> None:
        if is_improvement(value, self.best_value, self.mode, self.epsilon):
            self.best_value = value
            torch.save(checkpoint_dict, self.filepath)


class EarlyStopping:
    """
    EarlyStopping stops training when a monitored metric has not improved for a given number of epochs.

    Args:
    - patience: number of epochs to wait for an improvement before stopping
    - epsilon: minimum improvement to consider as an actual improvement (default: 0.0
    - mode: "min" or "max", whether the monitored metric should be minimized or maximized
    """

    def __init__(self, patience: int, epsilon: float = 1e-4, mode: str = "min") -> None:
        assert mode in {"min", "max"}

        self.patience = patience
        self.epsilon = epsilon
        self.mode = mode
        self.reset()

    def reset(self) -> None:
        self.best_value = float("inf") if self.mode == "min" else -float("inf")
        self.wait = 0

    def step(self, value: float) -> bool:
        if is_improvement(value, self.best_value, self.mode, self.epsilon):
            print(f"* metrics improved from {self.best_value:.6f} to {value:.6f}")
            self.best_value = value
            self.wait = 0
            return False
        else:
            print(f"metrics did not improve from {self.best_value:.6f}")

        self.wait += 1
        early_stop = self.wait >= self.patience
        if early_stop:
            print(f"Early stopping, no improvement in the last {self.patience} epochs")

        return early_stop


class ReduceLROnPlateau:
    """
    ReduceLROnPlateau reduces the learning rate when a monitored metric has stopped improving.

    Args:
    - factor: factor by which the learning rate will be reduced (new_lr = old_lr * factor)
    - patience: number of epochs to wait for an improvement before reducing the learning rate
    - cooldown: number of epochs to wait after reducing the learning rate before resuming normal operation
    - mode: "min" or "max", whether the monitored metric should be minimized or maximized
    - epsilon: minimum improvement to consider as an actual improvement (default: 1e-4)
    - min_lr: minimum learning rate after reduction (default: 1e-6)
    """

    def __init__(
        self,
        factor: float,
        patience: int,
        cooldown: int,
        mode: str = "min",
        epsilon: float = 1e-4,
        min_lr: float = 1e-6,
    ) -> None:
        assert mode in {"min", "max"}

        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.epsilon = epsilon
        self.min_lr = min_lr
        self.cooldown = cooldown
        self.reset()

    def reset(self) -> None:
        self.best_value = float("inf") if self.mode == "min" else -float("inf")
        self.wait = 0
        self.cooldown_counter = 0

    def in_cooldown(self) -> bool:
        return self.cooldown_counter > 0

    def step(self, value: float, optimizer_dict: Dict) -> None:
        if self.in_cooldown():
            self.cooldown_counter -= 1
            return

        if is_improvement(value, self.best_value, self.mode, self.epsilon):
            self.best_value = value
            self.wait = 0
            return

        self.wait += 1

        if self.wait >= self.patience:
            for optimizer in optimizer_dict.values():
                for group in optimizer["param_groups"]:
                    old_lr = group["lr"]
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    group["lr"] = new_lr

            self.wait = 0
            self.cooldown_counter = self.cooldown
            print(f"Reducing LR in the next epoch from {old_lr:.6f} to {new_lr:.6f}")


class TrainingCallback:
    """
    TrainingCallback is a class that manages all of the training callbacks.

    Args:
    - checkpoint: TrainCheckpoint instance for saving model checkpoints
    - early_stop: EarlyStopping instance for early stopping
    - lr_scheduler: ReduceLROnPlateau instance for learning rate scheduling
    """

    def __init__(
        self,
        checkpoint: Optional[TrainCheckpoint] = None,
        early_stop: Optional[EarlyStopping] = None,
        lr_scheduler: Optional[ReduceLROnPlateau] = None,
    ) -> None:
        self.stop_training = False
        self.checkpoint = checkpoint
        self.early_stop = early_stop
        self.lr_scheduler = lr_scheduler

    def reset(self) -> None:
        self.stop_training = False
        if self.checkpoint:
            self.checkpoint.reset()
        if self.early_stop:
            self.early_stop.reset()
        if self.lr_scheduler:
            self.lr_scheduler.reset()

    def is_training_stopped(self) -> bool:
        return self.stop_training

    def step(
        self,
        monitor_value: float,
        epoch: int,
        model_dict: Optional[Dict] = None,
        optimizer_dict: Optional[Dict] = None,
    ) -> bool:
        # training is stopped
        if self.stop_training:
            return True

        # model checkpoint
        if self.checkpoint and model_dict:
            checkpoint_dict = {"model": model_dict, "epoch": epoch}
            # only if optimizer is provided
            if optimizer_dict:
                checkpoint_dict["optimizer"] = {
                    k: v.state_dict() if hasattr(v, "state_dict") else v
                    for k, v in optimizer_dict.items()
                }
            print(checkpoint_dict)
            # save checkpoint
            self.checkpoint.step(monitor_value, checkpoint_dict)

        # lr scheduler
        if self.lr_scheduler and optimizer_dict:
            self.lr_scheduler.step(monitor_value, optimizer_dict)

        # early stopping
        if self.early_stop:
            # check if scheduler is in cooldown
            if self.lr_scheduler:
                if self.lr_scheduler.in_cooldown():
                    # skip early stopping check
                    return False
            # step
            stop_training = self.early_stop.step(monitor_value)
            self.stop_training = stop_training
            return stop_training

        # default behavior is to continue training
        return False


def time_formatter(sec_elapsed: float) -> str:
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m}:{round(s, 1)}"


def model_summary(config, tokenizer, depth: int):
    model = build_model(config, tokenizer)

    return summary(
        model,
        input_data=[
            torch.ones(
                config["batch_size"], config["estimated_seq_len"], dtype=torch.long
            ),  # x
            torch.ones(
                config["batch_size"],
                1,
                1,
                config["estimated_seq_len"],
                dtype=torch.bool,
            ),  # mask
        ],
        col_names=["input_size", "output_size", "num_params", "trainable"],
        depth=depth,
        row_settings=["var_names"],
    )
