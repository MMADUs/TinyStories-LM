# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import torch
from easydict import EasyDict

MoE_9M = EasyDict(__name__="Mixture of Experts 9M Params Config")

MoE_9M.random_seed = 42
MoE_9M.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)  # torch device
MoE_9M.init_std = 0.02  # parameter initialization standard deviation

# Decoder Configuration

MoE_9M.d_model = 256  # model dimension (dim of the vector embedding)
MoE_9M.num_heads = (
    8  # num of attention heads (must be divisor of d_model: d_model // h)
)
MoE_9M.num_layers = 4  # num of decoder layer (model depth)
MoE_9M.dropout = 0.1  # model dropout rate

# MoE configuration

MoE_9M.enable_moe = True  # mixture of experts flag
MoE_9M.n_experts = 4  # number of experts
MoE_9M.expert_d_ff = 256  # dim of the expert feed forward
MoE_9M.top_k_expert = 2  # top k most relevant expert to route
MoE_9M.aux_loss_coef = 0.01  # scale factor of aux loss, default is 0.01
