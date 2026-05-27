# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb


class LoRALinear(nn.Module):
    """LoRA (low rank adapation) layer for linear layers

    Args:
        base_layer: the linear layer
        rank: dimension of low-rank
        alpha: scaling factor
        dropout: dropout rate
        is_quantized: enable quantization on base layer weight
        compute_dtype: forward compute data type
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        # typical values, a multiplier of 2
        rank: int = 8,
        # common rule: rank * 2
        alpha: int = 16,
        dropout: float = 0.0,
        # unlock QLoRA
        is_quantized: bool = False,
        compute_dtype=torch.float16,
    ):
        super().__init__()

        assert rank > 0, "rank must be greater than 0"

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.is_quantized = is_quantized
        self.dropout = nn.Dropout(dropout)

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        if is_quantized:
            # bnb linear in 4bit
            self.base_layer = bnb.nn.Linear4bit(
                self.in_features,
                self.out_features,
                bias=base_layer.bias is not None,
                compute_dtype=compute_dtype,
                quant_type="nf4",
                compress_statistics=True,
            )

            # bnb linear weight in 4bit
            self.base_layer.weight = bnb.nn.Params4bit(
                base_layer.weight.data.clone(),
                requires_grad=False,
                compress_statistics=True,
                quant_type="nf4",
            )

            if base_layer.bias is not None:
                self.base_layer.bias = nn.Parameter(
                    base_layer.bias.data.clone(),
                    requires_grad=False,
                )
        else:
            self.base_layer = base_layer

            # freeze original base layer
            for p in self.base_layer.parameters():
                p.requires_grad = False

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, rank))

        # init LoRA
        nn.init.kaiming_uniform(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # freezed layer
        base_out = self.base_layer(x)
        lora_x = self.dropout(x)

        # lower rank layer
        lora_out = F.linear(input=lora_x, weight=self.lora_A, bias=None)
        lora_out = F.linear(input=lora_out, weight=self.lora_B, bias=None)

        return base_out + self.scaling * lora_out


# available modules for lora target modules
AVAILABLE_MODULES = [
    # attention projections
    "w_q",
    "w_k",
    "w_v",
    "w_o",
    # SwiGLU feed-forward projections
    "proj1",
    "proj2",
    # MoE router
    "gate",
]


def apply_lora(
    model: nn.Module,
    target_modules: tuple[str, ...] | None = None,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    is_quantized: bool = False,
    compute_dtype=torch.float16,
):
    """Apply LoRA to model linear layers

    Args:
        model: decoder model
        target_modules: target layer
        rank: dimension of low-rank
        alpha: scaling factor
        dropout: dropout rate
        is_quantized: enable quantization on base layer weight
        compute_dtype: forward compute data type

    Returns:
        model: injected lora model
    """
    if target_modules is None:
        return model

    for module_name, module in model.named_modules():
        for child_name, child_module in list(module.named_children()):
            # check if linear
            if isinstance(child_module, nn.Linear) and child_name in target_modules:
                # build lora linear
                lora = LoRALinear(
                    base_layer=child_module,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    is_quantized=is_quantized,
                    compute_dtype=compute_dtype,
                )
                # inject
                setattr(
                    obj=module,
                    name=child_name,
                    value=lora,
                )

    return model
