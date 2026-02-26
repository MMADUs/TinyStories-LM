# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import os
import time
import random
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tokenizers import Tokenizer

from dataset import get_dataloaders
from model import build_model, generate_causal_mask
from utils import TrainingCallback, TrainCheckpoint, EarlyStopping, time_formatter


def set_random_seed(seed: int):
    """
    Set random seed for reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_default_callbacks(config, version):
    """
    Default callback for TrainCheckpoint and EarlyStopping.

    Args:
    - config: model configuration dictionary
    - version: version string to identify the checkpoint file
    """
    output_dir = Path(config["output_dir_path"])
    ckpt_filename = config["model_ckpt_filename"].format(version)
    ckpt_path = output_dir / ckpt_filename

    # model checkpointing
    checkpoint_callback = TrainCheckpoint(
        filepath=ckpt_path,  # checkpoint file path
        epsilon=config["ckpt_epsilon"],  # minimum improvement to save a new checkpoint
        mode=config["callback_metrics_mode"],
    )

    # early stopping if model does not improve
    early_stopping_callback = EarlyStopping(
        patience=config[
            "early_stopping_patience"
        ],  # early stopping patience (in epochs)
        epsilon=config[
            "early_stopping_epsilon"
        ],  # minimum improvement to consider as an actual improvement
        mode=config["callback_metrics_mode"],
    )

    return TrainingCallback(
        checkpoint=checkpoint_callback,
        early_stop=early_stopping_callback,
    )


def get_last_checkpoint_state(config, tokenizer, version):
    """
    Load the last checkpoint state for model and optimizer. This is used to resume training from the last checkpoint.

    Args:
    - config: model configuration dictionary
    - tokenizer: trained tokenizer to build the model
    - version: version string to identify the checkpoint file
    """
    device = config["device"]

    output_dir = Path(config["output_dir_path"])
    ckpt_filename = config["model_ckpt_filename"].format(version)
    ckpt_path = output_dir / ckpt_filename

    checkpoint = torch.load(ckpt_path)
    model_state = checkpoint["model"]["decoder"]
    optimizer_state = checkpoint["optimizer"]["adamw"]
    last_epoch = checkpoint["epoch"]

    model = build_model(config, tokenizer)
    model.load_state_dict(model_state)
    model = model.to(device)

    lr = config["lr"]
    weight_decay = config["weight_decay"]

    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay)
    optimizer.load_state_dict(optimizer_state)

    return model, optimizer, last_epoch


def train_model(
    config,
    train_dl: DataLoader,
    val_dl: DataLoader,
    tokenizer: Tokenizer,
    callbacks: Optional[TrainingCallback] = None,
    initial_train: bool = False,
    version: str = "NA",
) -> dict:
    """
    The training loop.

    Args:
    - config: model configuration dictionary
    - train_dl: training dataloader
    - val_dl: validation dataloader
    - tokenizer: trained tokenizer
    - callbacks: TrainingCallback instance for handling training callbacks
    - initial_train: whether this is the initial training (default: False)
    - version: version string to identify the checkpoint file (default: "NA")
    """
    # set random seed for reproducibility
    set_random_seed(config["random_seed"])

    device = config["device"]

    output_dir = Path(config["output_dir_path"])
    ckpt_filename = config["model_ckpt_filename"].format(version)
    ckpt_path = output_dir / ckpt_filename

    init_epoch = 0

    # initialize model and optimizer
    if initial_train or not ckpt_path.exists():
        print("No checkpoint found, start initial training")

        model = build_model(config, tokenizer)
        model = model.to(device)

        lr = config["lr"]
        weight_decay = config["weight_decay"]

        optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay)
    else:
        print(f"Checkpoint found at {ckpt_path}, loading checkpoint for training")

        model, optimizer, last_epoch = get_last_checkpoint_state(
            config, tokenizer, version
        )
        init_epoch = last_epoch + 1

    # warmup + linear decay learning rate scheduler
    lr_warmup_percentage = config["lr_warmup_percentage"]
    remaining_epochs = config["num_epochs"] - init_epoch
    num_training_steps = len(train_dl) * remaining_epochs
    num_warmup_steps = int(lr_warmup_percentage * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # criterion
    label_smoothing = config["label_smoothing"]
    cross_entropy_ignore_index = config["cross_entropy_ignore_index"]

    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=cross_entropy_ignore_index,  # ignore index for padding
        label_smoothing=label_smoothing,  # label smoothing factor
    )

    # gradient scaler
    scaler = GradScaler()

    # init callbacks
    if callbacks is None:
        callbacks = get_default_callbacks(config, version)

    callbacks.reset()

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_perplexity": [],
        "val_perplexity": [],
    }

    start_time = time.time()

    for epoch in range(init_epoch, config["num_epochs"]):
        epoch_start = time.time()
        torch.cuda.empty_cache()

        model.train()
        train_loss = 0.0

        batch_iter = tqdm(train_dl, desc=f"epoch {epoch+1}")

        for batch in batch_iter:
            # input
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # attention mask
            padding_mask = batch["padding_mask"]
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # shape -> (B,1,1,L)
            causal_mask = generate_causal_mask(
                input_ids.size(1), device
            )  # shape -> (1,1,L,L)
            attention_mask = (
                padding_mask & causal_mask
            )  # combine padding and causal masks
            attention_mask = attention_mask.to(device)

            # reset optimizer gradient
            optimizer.zero_grad()

            with autocast(device_type=device):
                # forward
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                # compute loss
                loss = loss_fn(
                    logits.view(-1, tokenizer.get_vocab_size()), labels.view(-1)
                )

            # scale gradient then backward
            scaler.scale(loss).backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["clip_grad_max_norm"]
            )
            # step optimizer
            scaler.step(optimizer)
            # step scheduler
            scheduler.step()
            # update scaler
            scaler.update()
            # add train loss
            train_loss += loss.item()
            # show train info in tqdm
            batch_iter.set_postfix({"loss": f"{loss.item():6.3f}"})

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"validation"):
                # input
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                # attention mask
                padding_mask = batch["padding_mask"]
                padding_mask = padding_mask.unsqueeze(1).unsqueeze(
                    2
                )  # shape -> (B,1,1,L)
                causal_mask = generate_causal_mask(
                    input_ids.size(1), device
                )  # shape -> (1,1,L,L)
                attention_mask = (
                    padding_mask & causal_mask
                )  # combine padding and causal masks
                attention_mask = attention_mask.to(device)

                with autocast(device_type=device):
                    # forward
                    logits = model(input_ids=input_ids, attention_mask=attention_mask)
                    # compute loss
                    loss = loss_fn(
                        logits.view(-1, tokenizer.get_vocab_size()), labels.view(-1)
                    )

                val_loss += loss.item()

        # average losses
        train_loss /= len(train_dl)
        val_loss /= len(val_dl)

        # perplexity metric
        train_ppl = float(np.exp(train_loss))
        val_ppl = float(np.exp(val_loss))

        # history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_perplexity"].append(train_ppl)
        history["val_perplexity"].append(val_ppl)

        # logging
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch+1} - {time_formatter(epoch_time)} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"train_ppl={train_ppl:.2f} | val_ppl={val_ppl:.2f}"
        )

        # callbacks
        model_dict = {
            "decoder": model.state_dict(),
        }
        optimizer_dict = {
            "adamw": optimizer.state_dict(),
        }

        stop_training = callbacks.step(
            monitor_value=val_ppl,
            epoch=epoch,
            model_dict=model_dict,
            optimizer_dict=optimizer_dict,
        )

        if stop_training:
            break

        print("\n")

    end_time = time.time()
    print(f"elapsed time: {time_formatter(end_time - start_time)}")
    return history


def eval_model(config, version: str, test_dl: DataLoader, tokenizer: Tokenizer):
    """
    Evaluate the model on the test set using the last checkpoint.

    Args:
    - config: model configuration dictionary
    - version: version string to identify the checkpoint file
    - test_dl: test dataloader
    - tokenizer: trained tokenizer
    """
    device = config["device"]

    model, optimizer, _ = get_last_checkpoint_state(config, tokenizer, version)

    model.eval()
    test_loss = 0.0

    ignore_index = config["cross_entropy_ignore_index"]
    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=ignore_index,
    )

    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Evaluating"):
            # input
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            padding_mask = batch["padding_mask"].to(device)

            # attention mask
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,L)
            causal_mask = generate_causal_mask(input_ids.size(1), device)  # (1,1,L,L)
            attention_mask = padding_mask & causal_mask  # combine
            attention_mask = attention_mask.to(device)

            # forward pass
            logits = model(input_ids=input_ids, mask=attention_mask)

            # compute loss
            loss = loss_fn(logits.view(-1, tokenizer.get_vocab_size()), labels.view(-1))
            test_loss += loss.item()

    # average over batches
    test_loss /= len(test_dl)
    test_ppl = float(np.exp(test_loss))

    print(f"Test Loss: {test_loss:.6f} | Test PPL: {test_ppl:.2f}")
    return {"test_loss": test_loss, "test_perplexity": test_ppl}
