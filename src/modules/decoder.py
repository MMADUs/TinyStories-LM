# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import torch
import torch.nn as nn

from src.modules.blocks import (
    InputEmbedding,
    MultiHeadAttention,
    RMSNorm,
    ResidualConnection,
)
from src.modules.moe import SwiGLUFeedForward, MixtureOfExperts
from src.modules.utils import get_attn_mask, init_params


class DecoderBlock(nn.Module):
    """the decoder block

    norm -> masked attn -> residual -> norm -> ffn/moe -> residual

    Args:
        d_model: dimension of the model (embedding size)
        h: number of attention heads
        d_ff: dimension of the feed forward network
        dropout: dropout rate
        is_training: enable during training phase (specifically for MoE)
        using_moe: mixture of experts flag
        num_experts: number of MoE experts
        top_k: top k experts to route
        aux_loss_coef: auxiliary loss scale factor
    """

    def __init__(
        self,
        d_model: int,
        h: int,
        d_ff: int,
        dropout: float,
        # required args to enable MoE, default to feedforward
        is_training: bool,
        expert_d_ff: int,
        using_moe: bool = False,
        num_experts: int = None,
        top_k: int = None,
        aux_loss_coef: float = None,
    ):
        super().__init__()

        self.using_moe = using_moe

        self.attn = MultiHeadAttention(d_model, h, dropout)

        if using_moe:
            self.ffn = MixtureOfExperts(
                d_model,
                expert_d_ff,
                dropout,
                is_training,
                num_experts,
                top_k,
                aux_loss_coef,
            )
        else:
            self.ffn = SwiGLUFeedForward(d_model, d_ff, dropout)

        # residual blocks
        self.attn_resid = ResidualConnection(d_model, dropout)
        self.ffn_resid = ResidualConnection(d_model, dropout)

    def forward(self, x, mask):
        # masked attention
        x = self.attn_resid(
            x,
            lambda norm_x: self.attn(norm_x, mask),
        )

        # feed forward / MoE
        if self.using_moe:
            temp = {}

            def moe_sublayer(norm_x):
                moe_out, aux_loss = self.ffn(norm_x)
                temp["aux_loss"] = aux_loss
                return moe_out

            x = self.ffn_resid(x, moe_sublayer)
            aux_loss = temp["aux_loss"]
        else:
            x = self.ffn_resid(x, self.ffn)
            aux_loss = x.new_tensor(0.0)

        return x, aux_loss


class LMDecoder(nn.Module):
    """the decoder model class

    Args:
        vocab_size: size of the vocabulary
        d_model: dimension of the model (embedding size)
        h: number of attention heads
        d_ff: dimension of the feed forward network
        num_layers: number of decoder blocks
        dropout: dropout rate
        is_training: enable during training phase (specifically for MoE)
        using_moe: mixture of experts flag
        num_experts: number of MoE experts
        top_k: top k experts to route
        aux_loss_coef: auxiliary loss scale factor
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        h: int,
        d_ff: int,
        num_layers: int,
        dropout: float,
        # required args to enable MoE, default to feedforward
        using_moe: bool,
        is_training: bool,
        num_experts: int,
        expert_d_ff: int,
        top_k: int,
        aux_loss_coef: float,
    ):
        super().__init__()

        self.embedding = InputEmbedding(d_model, vocab_size)  # token embedding
        # stack of N decoder blocks
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    # std config
                    d_model=d_model,
                    h=h,
                    d_ff=d_ff,
                    dropout=dropout,
                    # moe config
                    using_moe=using_moe,
                    is_training=is_training,
                    expert_d_ff=expert_d_ff,
                    num_experts=num_experts,
                    top_k=top_k,
                    aux_loss_coef=aux_loss_coef,
                )
                for _ in range(num_layers)
            ]
        )

        # final norm layer
        self.norm = RMSNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size, bias=False)  # vocab projection

        self.proj.weight = self.embedding.get_weights()  # weight tying

    def forward(self, x, mask):
        # token to embedding
        x = self.embedding(x)

        total_aux_loss = x.new_tensor(0.0)

        # iterate through decoder blocks
        for layer in self.layers:
            x, aux_loss = layer(x, mask)
            total_aux_loss += aux_loss

        # normalize and project to vocab size
        x = self.norm(x)
        logits = self.proj(x)

        return logits, total_aux_loss

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
        """autoregressive generation: currently only support 1 batch"""

        pad_id = special_tokens["padding"]
        eos_id = special_tokens["end"]

        for _ in range(max_new_tokens):
            attention_mask = get_attn_mask(input_ids, pad_id).to(input_ids.device)
            logits, _ = self(input_ids, attention_mask)

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


def build_model(config, tokenizer, is_training):
    """helper function

    Args:
        config: a dictionary containing model configuration
        tokenizer: a tokenizer object with a vocab_size attribute
        is_training: enable during training phase (specifically for MoE)
    """

    # build decoder model
    model = LMDecoder(
        # mandatory config
        vocab_size=tokenizer.get_vocab_size(),
        d_model=config["d_model"],
        h=config["num_heads"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        using_moe=config.get("enable_moe", False),  # ffn is default
        # place config depending on 'enable_moe'
        # safely ignored when using MoE
        d_ff=config.get("d_ff", None),
        is_training=is_training,
        # safely ignored when using d_ff
        num_experts=config.get("n_experts", None),
        expert_d_ff=config.get("expert_d_ff", None),
        top_k=config.get("top_k_expert", None),
        aux_loss_coef=config.get("aux_loss_coef", 0.01),
    )

    # initialize weights
    init_params(
        model,
        init_std=config["init_std"],
        n_layers=config["num_layers"],
    )

    return model
