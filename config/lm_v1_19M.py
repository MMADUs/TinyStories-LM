# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import torch
from easydict import EasyDict

_19M_config = EasyDict(__name__="19M Params Model Configuration")

_19M_config.random_seed = 42
_19M_config.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)  # torch device
_19M_config.init_std = 0.02  # parameter initialization standard deviation

_19M_config.num_layers = 6  # num of decoder layer (model depth)
_19M_config.num_heads = (
    8  # num of attention heads (must be divisor of d_model: d_model // h)
)
_19M_config.d_model = 256  # model dimension (dim of the vector embedding)
_19M_config.d_ff = 1024  # dim of the feed forward block (2-4 * d_model)
_19M_config.dropout = 0.1  # model dropout rate
