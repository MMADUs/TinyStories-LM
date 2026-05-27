# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import torch
from easydict import EasyDict

MoE_43M = EasyDict(__name__="Mixture of Experts 43M Params Config")

MoE_43M.random_seed = 42
MoE_43M.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)  # torch device
MoE_43M.init_std = 0.02  # parameter initialization standard deviation

# Decoder Configuration

MoE_43M.d_model = 512  # model dimension (dim of the vector embedding)
MoE_43M.num_heads = (
    8  # num of attention heads (must be divisor of d_model: d_model // h)
)
MoE_43M.num_layers = 8  # num of decoder layer (model depth)
MoE_43M.dropout = 0.1  # model dropout rate

# MoE configuration

MoE_43M.enable_moe = True  # mixture of experts flag
MoE_43M.n_experts = 4  # number of experts
MoE_43M.expert_d_ff = 512  # dim of the expert feed forward
MoE_43M.top_k_expert = 2  # top k most relevant expert to route
MoE_43M.aux_loss_coef = 0.01  # scale factor of aux loss, default is 0.01
