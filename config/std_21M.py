# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import torch
from easydict import EasyDict

std_21M = EasyDict(__name__="Standard 21M Params Config")

std_21M.random_seed = 42
std_21M.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)  # torch device
std_21M.init_std = 0.02  # parameter initialization standard deviation

# Decoder Configuration

std_21M.d_model = 384  # model dimension (dim of the vector embedding)
std_21M.num_heads = (
    8  # num of attention heads (must be divisor of d_model: d_model // h)
)
std_21M.num_layers = 6  # num of decoder layer (model depth)
std_21M.d_ff = 1536  # dim of the feed forward block (2-4 * d_model)
std_21M.dropout = 0.1  # model dropout rate

std_21M.enable_moe = False  # mixture of experts flag
