# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import torch
from easydict import EasyDict

std_9M = EasyDict(__name__="Standard 9M Params Config")

std_9M.random_seed = 42
std_9M.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)  # torch device
std_9M.init_std = 0.02  # parameter initialization standard deviation

# Decoder Configuration

std_9M.d_model = 256  # model dimension (dim of the vector embedding)
std_9M.num_heads = (
    8  # num of attention heads (must be divisor of d_model: d_model // h)
)
std_9M.num_layers = 4  # num of decoder layer (model depth)
std_9M.d_ff = 1024  # dim of the feed forward block (2-4 * d_model)
std_9M.dropout = 0.1  # model dropout rate

std_9M.enable_moe = False  # mixture of experts flag
