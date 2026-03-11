# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class InputEmbedding(nn.Module):
    """
    Input embedding layer that converts token indices to dense vectors.

    Args:
    - d_model: dimension of the model (embedding size)
    - vocab_size: size of the vocabulary
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


def rotate_half(x):
    """
    Rotate every pair of elements in the last dimension by 90 degrees.
    """
    # split the last dim, x: (..., d)
    x1 = x[..., ::2]  # even indices
    x2 = x[..., 1::2]  # odd indices
    # 90 degree rotation: (x1, x2) --> (-x2, x1)
    x_rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return x_rot


def apply_RoPE(q, k, base=10000):
    """
    Apply RoPE (rotary positional embeddings) to query and key.
    q, k: (batch, heads, seq_len, d_k)
    """
    seq_len = q.shape[2]
    d = q.shape[-1]
    device = q.device

    # position indices
    pos = torch.arange(seq_len, device=device).float()  # (seq_len)
    # sinusoidal frequencies
    theta = 1.0 / (base ** (torch.arange(0, d, 2).float() / d)).to(device)  # (d/2)

    # the rotation angle (in radians) for each sequence position
    freq = torch.einsum("i,j->ij", pos, theta)  # (seq_len, d/2)

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
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin

    return q_rot, k_rot


class MultiHeadAttention(nn.Module):
    """
    A multi-head self-attention with RoPE.

    Args:
    - d_model: dimension of the model (embedding size)
    - h: number of attention heads
    - dropout: dropout rate
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
    def attention(query, key, value, dropout: nn.Dropout, mask=None):
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


class SwiGLUFeedForward(nn.Module):
    """
    A feed forward network with SwiGLU activation.
    Flow: linear --> SwiGLU --> linear --> dropout.

    Args:
    - d_model: dimension of the model (embedding size)
    - d_ff: dimension of the feed forward network
    - dropout: dropout rate
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()

        self.proj1 = nn.Linear(d_model, d_ff * 2)  # gating
        self.proj2 = nn.Linear(d_ff, d_model)  # output projection

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, d_model)

        # x1, x2: (batch, seq_len, d_ff)
        x1, x2 = self.proj1(x).chunk(2, dim=-1)

        # SwiGLU activation
        x = F.silu(x1) * x2

        # out: (batch, seq_len, d_model)
        out = self.proj2(x)
        return self.dropout(out)


class LayerNormalization(nn.Module):
    """
    A layer normalization that normalizes across the last dimension (d_model).
    Stabilizes training and better gradient flow.

    Args:
    - d_model: dimension of the model (embedding size)
    - eps: small value to prevent division by zero (default: 1e-6)
    """

    def __init__(self, d_model: int, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        # alpha and bias are learnable parameters
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # x: (batch, seq_len, d_model)

        # mean, std: (batch, seq_len, 1)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        # y = a * (x - mean) / (std + eps) + b
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class RMSNorm(nn.Module):
    """
    A RMS normalization that normalizes across the last dimension (d_model) using root mean square.
    A simpler alternative to layer normalization.

    Args:
    - d_model: dimension of the model (embedding size)
    - eps: small value to prevent division by zero (default: 1e-6)
    """

    def __init__(self, d_model: int, eps: float = 10**-6):
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
    """
    A residual connection with pre-layer normalization.
    Ensure gradient flow = prevent vanishing gradients.
    Flow: norm --> sublayer --> dropout --> residual add.

    Args:
    - d_model: dimension of the model (embedding size)
    - dropout: dropout rate
    - norm_strategy: normalization strategy (RMSNorm or LayerNorm)
    """

    def __init__(self, d_model: int, dropout: float, norm_strategy: str):
        super().__init__()
        self.norm_strategy = norm_strategy

        # choose normalization strategy based on config
        if norm_strategy == "RMSNorm":
            self.norm = RMSNorm(d_model)
        elif norm_strategy == "LayerNorm":
            self.norm = LayerNormalization(d_model)
        else:
            raise ValueError(f"invalid normalization strategy: {norm_strategy}")

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # pre-norm: apply lnorm before the sublayer
        norm = self.norm(x)
        x_hat = sublayer(norm)
        out = self.dropout(x_hat)

        # residual connection
        return x + out


class DecoderBlock(nn.Module):
    """
    A single block of the decoder.
    Flow: norm --> masked attn --> residual --> norm --> ffn --> residual.

    Args:
    - d_model: dimension of the model (embedding size)
    - h: number of attention heads
    - d_ff: dimension of the feed forward network
    - dropout: dropout rate
    - norm_strategy: normalization strategy (RMSNorm or LayerNorm)
    """

    def __init__(
        self,
        d_model: int,
        h: int,
        d_ff: int,
        dropout: float,
        norm_strategy: str,
    ):
        super().__init__()

        self.attn = MultiHeadAttention(d_model, h, dropout)
        self.ffn = SwiGLUFeedForward(d_model, d_ff, dropout)
        # residual blocks
        self.attn_resid = ResidualConnection(d_model, dropout, norm_strategy)
        self.ffn_resid = ResidualConnection(d_model, dropout, norm_strategy)

    def forward(self, x, mask):
        # masked attention
        x = self.attn_resid(x, lambda x: self.attn(x, mask))
        # feed forward
        x = self.ffn_resid(x, self.ffn)
        return x


class LMDecoder(nn.Module):
    """
    The Decoder Model.

    Args:
    - vocab_size: size of the vocabulary
    - d_model: dimension of the model (embedding size)
    - h: number of attention heads
    - d_ff: dimension of the feed forward network
    - num_layers: number of decoder blocks
    - dropout: dropout rate
    - norm_strategy: normalization strategy (RMSNorm by default)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        h: int,
        d_ff: int,
        num_layers: int,
        dropout: float,
        norm_strategy: str = "RMSNorm",
    ):
        super().__init__()

        self.embedding = InputEmbedding(d_model, vocab_size)  # token embedding
        # stack of N decoder blocks
        self.layers = nn.ModuleList(
            [
                DecoderBlock(d_model, h, d_ff, dropout, norm_strategy)
                for _ in range(num_layers)
            ]
        )
        self.norm = LayerNormalization(d_model)  # final layer normalization
        self.proj = nn.Linear(d_model, vocab_size, bias=False)  # vocab projection

    def forward(self, x, mask):
        # token to embedding
        x = self.embedding(x)

        # iterate through decoder blocks
        for layer in self.layers:
            x = layer(x, mask)

        # normalize and project to vocab size
        x = self.norm(x)
        logits = self.proj(x)
        return logits


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
    causal = generate_causal_mask(input_ids.size(1), input_ids.device)  # (1, 1, seq_len, seq_len)
    padding = generate_padding_mask(input_ids, pad_id) # (B, 1, 1, L)
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


def build_model(config, tokenizer):
    """
    Build the MicroLM model from the configuration and tokenizer.

    Args:
    - config: a dictionary containing model hyperparameters
    - tokenizer: a tokenizer object with a vocab_size attribute

    Returns:
    - model: an instance of the MicroLM class
    """
    vocab_size = tokenizer.get_vocab_size()
    d_model = config["d_model"]
    h = config["num_heads"]
    d_ff = config["d_ff"]
    num_layers = config["num_layers"]
    dropout = config["dropout"]
    norm_strategy = config["norm_strategy"]
    init_std = config["init_std"]

    model = LMDecoder(vocab_size, d_model, h, d_ff, num_layers, dropout, norm_strategy)
    initialize_parameters(model, init_std)

    return model


def test_decoder():
    torch.manual_seed(0)
    vocab_size = 100
    d_model = 64
    h = 4
    d_ff = 256
    num_layers = 2
    dropout = 0.1
    batch_size = 2
    seq_len = 10

    model = LMDecoder(vocab_size, d_model, h, d_ff, num_layers, dropout)

    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    mask = generate_causal_mask(seq_len, x.device)

    logits = model(x, mask)

    assert logits.shape == (
        batch_size,
        seq_len,
        vocab_size,
    ), f"wrong shape, expected {(batch_size, seq_len, vocab_size)}, got {logits.shape}"

    print("decoder test passed")


if __name__ == "__main__":
    test_decoder()
