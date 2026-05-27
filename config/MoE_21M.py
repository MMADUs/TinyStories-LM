# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import torch
from easydict import EasyDict

MoE_21M = EasyDict(__name__="Mixture of Experts 21M Params Config")

MoE_21M.random_seed = 42
MoE_21M.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)  # torch device
MoE_21M.init_std = 0.02  # parameter initialization standard deviation

# Decoder Configuration

MoE_21M.d_model = 384  # model dimension (dim of the vector embedding)
MoE_21M.num_heads = (
    8  # num of attention heads (must be divisor of d_model: d_model // h)
)
MoE_21M.num_layers = 6  # num of decoder layer (model depth)
MoE_21M.dropout = 0.1  # model dropout rate

# MoE configuration

MoE_21M.enable_moe = True  # mixture of experts flag
MoE_21M.n_experts = 4  # number of experts
MoE_21M.expert_d_ff = 384  # dim of the expert feed forward
MoE_21M.top_k_expert = 2  # top k most relevant expert to route
MoE_21M.aux_loss_coef = 0.01  # scale factor of aux loss, default is 0.01
