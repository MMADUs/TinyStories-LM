# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import math
import torch
import torch.nn as nn

from src.modules.rope import apply_RoPE


class InputEmbedding(nn.Module):
    """input embedding layer.

    Args:
        d_model: dimension of the model (embedding size)
        vocab_size: size of the vocabulary
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def get_weights(self):
        return self.embedding.weight

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


class MultiHeadAttention(nn.Module):
    """multi-head self-attention blocks with RoPE.

    Args:
        d_model: dimension of the model (embedding size)
        h: number of attention heads
        dropout: dropout rate
    """

    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()

        # make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_model = d_model  # embedding vector size
        self.h = h  # number of heads
        self.d_k = d_model // h  # dimension of vector seen by each head

        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq (query)
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk (key)
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv (value)
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Wo (output)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dropout: nn.Dropout,
        mask=None,
    ):
        d_k = query.shape[-1]

        # compute attention score
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # apply masked attention
        if mask is not None:
            # mark with -inf to the position where mask == False
            # this way softmax will make the value 0
            scores = (
                scores.float()
            )  # safety for mixed precision training, inf requires float32
            scores.masked_fill_(mask == False, float("-inf"))
            scores = scores.to(query.dtype)

        # compute attention weight using softmax
        attn_w = torch.softmax(scores, dim=-1)

        if dropout is not None:
            attn_w = dropout(attn_w)

        # weighted sum of values
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        out = attn_w @ value

        return out, attn_w

    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model)
        batch, seq_len, _ = x.size()

        # query, key, value projections
        query = self.w_q(x)
        key = self.w_k(x)
        value = self.w_v(x)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(batch, seq_len, self.h, self.d_k).transpose(1, 2)
        key = key.view(batch, seq_len, self.h, self.d_k).transpose(1, 2)
        value = value.view(batch, seq_len, self.h, self.d_k).transpose(1, 2)

        # apply RoPE to query and key
        query, key = apply_RoPE(query, key)

        # masked attention
        out, _attn_w = self.attention(query, key, value, self.dropout, mask)

        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

        # multiply by Wo
        return self.w_o(out)


class RMSNorm(nn.Module):
    """normalization that normalizes using root mean square

    Args:
        d_model: dimension of the model (embedding size)
        eps: small value to prevent division by zero (default: 1e-6)
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # alpha is the only learnable parameter
        self.alpha = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # x: (batch, seq_len, d_model)

        # rms: (batch, seq_len, 1)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        # y = a * x / rms
        return self.alpha * x / rms


class ResidualConnection(nn.Module):
    """residual connection with pre-layer normalization

    Args:
        d_model: dimension of the model (embedding size)
        dropout: dropout rate
    """

    def __init__(self, d_model: int, dropout: float):
        super().__init__()

        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # pre-norm: apply norm before the sublayer
        norm = self.norm(x)
        x_hat = sublayer(norm)
        out = self.dropout(x_hat)

        # residual connection
        return x + out
