# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import torch
from easydict import EasyDict

std_43M = EasyDict(__name__="Standard 43M Params Config")

std_43M.random_seed = 42
std_43M.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)  # torch device
std_43M.init_std = 0.02  # parameter initialization standard deviation

# Decoder Configuration

std_43M.d_model = 512  # model dimension (dim of the vector embedding)
std_43M.num_heads = (
    8  # num of attention heads (must be divisor of d_model: d_model // h)
)
std_43M.num_layers = 8  # num of decoder layer (model depth)
std_43M.d_ff = 2048  # dim of the feed forward block (2-4 * d_model)
std_43M.dropout = 0.1  # model dropout rate

std_43M.enable_moe = False  # mixture of experts flag
