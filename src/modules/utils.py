# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import torch
import torch.nn as nn


def generate_causal_mask(seq_len, device):
    """
    Generate a causal mask for self-attention.
    """
    triangular = torch.tril(
        torch.ones((seq_len, seq_len), device=device, dtype=torch.bool)
    )
    # reshaping
    triangular = triangular.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    return triangular


def generate_padding_mask(input_ids, pad_id):
    """
    Generate a padding mask for self-attention.
    """
    mask = input_ids != pad_id  # True for valid tokens, False for padding
    # reshaping
    mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
    return mask


def get_attn_mask(input_ids, pad_id):
    causal = generate_causal_mask(
        input_ids.size(1), input_ids.device
    )  # (1, 1, seq_len, seq_len)
    padding = generate_padding_mask(input_ids, pad_id)  # (B, 1, 1, L)
    return causal & padding  # combine masks


def initialize_parameters(model, init_std):
    """
    Initialize the parameters of the model.

    Args:
    - model: the model to initialize
    - init_std: standard deviation of the Gassian distribution for weight initialization
    """
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.normal_(p, mean=0.0, std=init_std)  # weight matrices
        else:
            nn.init.zeros_(p)  # biases

    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
