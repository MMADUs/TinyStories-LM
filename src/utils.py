# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchinfo import summary

from src.modules.decoder import build_model


def set_random_seed(seed: int):
    """set random seed for reproducibility"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DeviceDataLoader:
    """wrapper around torch DataLoader that moves data to the specified device

    Args:
        dl: torch DataLoader to wrap
        device: torch device to move data to
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


def model_summary(config, model, batch_size, depth: int):
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
