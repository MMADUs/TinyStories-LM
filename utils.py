# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchinfo import summary

from model import build_model


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


def get_last_checkpoint_state(
    config, train_config, tokenizer, version, load_from_epoch
):
    """
    Load the last checkpoint state for model and optimizer. This is used to resume training from the last checkpoint.

    Args:
    - config: model configuration dictionary
    - train_config: training configuration dictionary
    - tokenizer: trained tokenizer to build the model
    - version: version string to identify the checkpoint file
    - load_from_epoch: epoch number to resume training from (default: None)
    """
    device = config["device"]

    output_dir = Path(config["output_dir_path"])
    filename = train_config["model_ckpt_filename"].format(version)  # base filename

    ckpt_filename = f"{filename}_epoch_{load_from_epoch}"  # epoch filename
    ckpt_filename = (
        ckpt_filename + train_config["ckpt_format"]
    )  # full filename with format
    ckpt_path = output_dir / ckpt_filename

    checkpoint = torch.load(ckpt_path)
    model_state = checkpoint["model"]
    optimizer_state = checkpoint["optimizer"]
    last_epoch = checkpoint["epoch"]

    val_loss = checkpoint["val_loss"]
    val_ppl = checkpoint["val_perplexity"]

    print(
        f"Checkpoint loaded from epoch {last_epoch+1} with val loss {val_loss:.4f} and val perplexity {val_ppl:.4f}"
    )

    model = build_model(config, tokenizer)
    model.load_state_dict(model_state)
    model = model.to(device)
    # model = torch.compile(model)

    lr = train_config["lr"]
    weight_decay = train_config["weight_decay"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, fused=True)
    optimizer.load_state_dict(optimizer_state)

    return model, optimizer, last_epoch


class DeviceDataLoader:
    """
    DeviceDataLoader is a wrapper around torch DataLoader that moves data to the specified device.

    Args:
    - dl: torch DataLoader to wrap
    - device: torch device to move data to
    """

    def __init__(self, dl: DataLoader, device):
        self.dl = dl
        self.device = device

    def _to_device(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        else:
            return batch  # leave other types untouched

    def __iter__(self):
        for batch in self.dl:
            yield self._to_device(batch)

    def __len__(self):
        return len(self.dl)


def time_formatter(sec_elapsed: float) -> str:
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m}:{round(s, 1)}"


def model_summary(config, batch_size, tokenizer, depth: int):
    model = build_model(config, tokenizer)

    # make sure weight is tied
    print(model.proj.weight is model.embedding.get_weights())

    # actual param count
    total = sum(p.numel() for p in model.parameters())
    print(f"total: {total/1e6:.2f}M")

    return summary(
        model,
        input_data=[
            torch.ones(batch_size, config["estimated_seq_len"], dtype=torch.long),  # x
            torch.ones(
                batch_size,
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
