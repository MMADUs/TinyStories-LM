# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """LoRA (low rank adapation) layer for linear layers

    Args:
        base_layer: the linear layer
        rank: dimension of low-rank
        alpha: scaling factor
        dropout: dropout rate
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        # typical values, a multiplier of 2
        rank: int = 8,
        # common rule: rank * 2
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))

        # init LoRA
        nn.init.kaiming_uniform(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # freeze original base layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

    def forward(self, x):
        # freezed layer
        x = self.base_layer(x)
        base_out = self.dropout(x)

        # lower rank layer
        lora_out = F.linear(input=self.base_layer, weight=self.lora_A, bias=None)
        lora_out = F.linear(input=lora_out, weight=self.lora_B, bias=None)

        return base_out + self.scaling * lora_out


def apply_lora(
    model: nn.Module,
    target_modules: tuple[str, ...] | None = None,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
):
    """Apply LoRA to model linear layers

    Args:
        model: decoder model
        target_modules: target layer
        rank: dimension of low-rank
        alpha: scaling factor
        dropout: dropout rate

    Returns:
        model: injected lora model
    """
    for module_name, module in model.named_parameters():
        for child_name, child_module in list(module.named_children()):
            # check if linear
            if isinstance(child_module, nn.Linear) and child_name in target_modules:
                lora = LoRALinear(
                    base_layer=child_module,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                setattr(
                    obj=module,
                    name=child_name,
                    value=lora,
                )

    return model
