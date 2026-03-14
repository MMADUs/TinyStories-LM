# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import torch
from easydict import EasyDict

_9M_config = EasyDict(__name__="9M Params Model Configuration")

_9M_config.random_seed = 42
_9M_config.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)  # torch device
_9M_config.init_std = 0.02  # parameter initialization standard deviation

_9M_config.num_layers = 8  # num of decoder layer (model depth)
_9M_config.num_heads = (
    8  # num of attention heads (must be divisor of d_model: d_model // h)
)
_9M_config.d_model = 128  # model dimension (dim of the vector embedding)
_9M_config.d_ff = 512  # dim of the feed forward block (2-4 * d_model)
_9M_config.dropout = 0.1  # model dropout rate
_9M_config.norm_strategy = "RMSNorm"  # normalization strategy (RMSNorm or LayerNorm)