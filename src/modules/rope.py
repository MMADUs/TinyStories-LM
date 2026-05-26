# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import torch


def rotate_half(x: torch.Tensor):
    """rotate every pair of elements in the last dimension by 90 degrees"""
    # split the last dim, x: (..., d)
    x1 = x[..., ::2]  # even indices
    x2 = x[..., 1::2]  # odd indices

    # 90 degree rotation: (x1, x2) --> (-x2, x1)
    x_rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return x_rot


def apply_RoPE(q: torch.Tensor, k: torch.Tensor, base=10000):
    """apply RoPE (rotary positional embeddings) to query and key

    Args:
        q: query (batch, heads, seq_len, d_k)
        k: key (batch, heads, seq_len, d_k)
        base: RoPE base frequency

    Returns:
        q_rot, k_rot: (batch, heads, seq_len, d_k)
    """
    seq_len = q.shape[2]
    d = q.shape[-1]
    device = q.device

    # position indices
    pos = torch.arange(seq_len, device=device).float()  # (seq_len)

    # sinusoidal frequencies
    theta = 1.0 / (base ** (torch.arange(0, d, 2).float() / d)).to(device)  # (d/2)

    # the rotation angle (in radians) for each position
    freq = pos[:, None] * theta[None, :]  # (seq_len, d/2)

    # sin, cos: (seq_len, d/2)
    sin = freq.sin()
    cos = freq.cos()

    # sin, cos: (seq_len, d)
    sin = torch.repeat_interleave(sin, 2, dim=-1)
    cos = torch.repeat_interleave(cos, 2, dim=-1)

    # sin, cos: (1, 1, seq_len, d)
    sin = sin.unsqueeze(0).unsqueeze(0)
    cos = cos.unsqueeze(0).unsqueeze(0)

    # apply RoPE to query and key
    # q_rot, k_rot: (batch, heads, seq_len, d_k)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin

    return q_rot, k_rot
