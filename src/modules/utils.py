# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import math

import torch
import torch.nn as nn

from src.modules.blocks import RMSNorm


def generate_causal_mask(seq_len, device):
    """generate a causal mask for self-attention
    
    Args:
        seq_len: length of sequence
        device: torch device
    
    Returns:
        cau_mask: (1, 1, seq_len, seq_len)
    """
    triangular = torch.tril(
        torch.ones((seq_len, seq_len), device=device, dtype=torch.bool)
    )
    # reshaping
    triangular = triangular.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    return triangular


def generate_padding_mask(input_ids, pad_id):
    """generate a padding mask for self-attention
    
    Args:
        input_ids: input tokens
        pad_id: padding token id
    
    Returns:
        pad_mask: (B, 1, 1, L)
    """
    mask = input_ids != pad_id  # True for valid tokens, False for padding
    # reshaping
    mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
    return mask


def get_attn_mask(input_ids, pad_id):
    """get attention mask for training

    Args:
        input_ids: input tokens
        pad_id: padding token id

    Returns:
        full_mask: (B, 1, 1, L)
    """
    causal = generate_causal_mask(
        input_ids.size(1), input_ids.device
    )  # (1, 1, seq_len, seq_len)
    padding = generate_padding_mask(input_ids, pad_id)  # (B, 1, 1, L)
    return causal & padding  # combine masks


def init_params(model: nn.Module, init_std, n_layers):
    """initialize model parameters

    Args:
        model: the model to initialize
        init_std: standard deviation
        n_layers: number of decoder blocks
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            std = init_std
            # smaller std for projection layers
            if name.endswith(("w_o", "proj_2")):
                std = init_std / math.sqrt(2 * n_layers)
            # layer weights
            nn.init.normal(module.weight, mean=0.0, std=std)
            # layer with bias
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # embedding init
        elif isinstance(module, nn.Embedding):
            nn.init.normal(module.weight, mean=0.0, std=std)
        # rms norm init
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.alpha)
