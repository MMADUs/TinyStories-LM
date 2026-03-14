# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import torch
from easydict import EasyDict

_40M_config = EasyDict(__name__="40M Params Model Configuration")

_40M_config.random_seed = 42
_40M_config.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)  # torch device
_40M_config.init_std = 0.02  # parameter initialization standard deviation

_40M_config.num_layers = 8  # num of decoder layer (model depth)
_40M_config.num_heads = (
    8  # num of attention heads (must be divisor of d_model: d_model // h)
)
_40M_config.d_model = 512  # model dimension (dim of the vector embedding)
_40M_config.d_ff = 2048  # dim of the feed forward block (2-4 * d_model)
_40M_config.dropout = 0.1  # model dropout rate
_40M_config.norm_strategy = "RMSNorm"  # normalization strategy (RMSNorm or LayerNorm)
