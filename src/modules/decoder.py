# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import torch
import torch.nn as nn

from src.modules.blocks import (
    InputEmbedding,
    MultiHeadAttention,
    RMSNorm,
    ResidualConnection,
)
from src.modules.moe import SwiGLUFeedForward
from src.modules.utils import get_attn_mask, initialize_parameters


class DecoderBlock(nn.Module):
    """
    A single block of the decoder.
    Flow: norm --> masked attn --> residual --> norm --> ffn --> residual.

    Args:
    - d_model: dimension of the model (embedding size)
    - h: number of attention heads
    - d_ff: dimension of the feed forward network
    - dropout: dropout rate
    """

    def __init__(
        self,
        d_model: int,
        h: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()

        self.attn = MultiHeadAttention(d_model, h, dropout)
        self.ffn = SwiGLUFeedForward(d_model, d_ff, dropout)
        # residual blocks
        self.attn_resid = ResidualConnection(d_model, dropout)
        self.ffn_resid = ResidualConnection(d_model, dropout)

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
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        h: int,
        d_ff: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()

        self.embedding = InputEmbedding(d_model, vocab_size)  # token embedding
        # stack of N decoder blocks
        self.layers = nn.ModuleList(
            [DecoderBlock(d_model, h, d_ff, dropout) for _ in range(num_layers)]
        )

        # final norm layer
        self.norm = RMSNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size, bias=False)  # vocab projection

        self.proj.weight = self.embedding.get_weights()  # weight tying

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

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        special_tokens,
        max_new_tokens,
        temperature=1.0,
        top_k=0,
        top_p=0.0,
    ):
        """
        Generate new tokens autoregressively given initial input_ids.
        """
        pad_id = special_tokens["padding"]
        eos_id = special_tokens["end"]

        for _ in range(max_new_tokens):
            attention_mask = get_attn_mask(input_ids, pad_id).to(input_ids.device)
            logits = self(input_ids, attention_mask)
            next_token_logits = logits[:, -1, :]  # (1, vocab_size)

            # apply temperature
            next_token_logits = next_token_logits / temperature

            # top-k filtering
            if top_k > 0:
                values, indices = torch.topk(next_token_logits, top_k, dim=-1)
                filtered = torch.full_like(next_token_logits, float("-inf"))
                filtered.scatter_(-1, indices, values)
                next_token_logits = filtered

            # top-p (nucleus) filtering
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p

                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()

                sorted_indices_to_remove[..., 0] = False

                sorted_logits = sorted_logits.masked_fill(
                    sorted_indices_to_remove, float("-inf")
                )

                next_token_logits = torch.full_like(next_token_logits, float("-inf"))
                next_token_logits.scatter_(
                    dim=-1, index=sorted_indices, src=sorted_logits
                )

            # sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, 1)

            # append
            input_ids = torch.cat([input_ids, next_token_id], dim=1)

            # stop if EOS is generated
            if next_token_id.item() == eos_id:
                break

        return input_ids


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
    init_std = config["init_std"]

    model = LMDecoder(vocab_size, d_model, h, d_ff, num_layers, dropout)
    initialize_parameters(model, init_std)

    return model