# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

from easydict import EasyDict

from config.lm_v1_19M import _19M_config as default_model_config
from config.dataset import dataset_config, tokenizer_config
from config.training import sft_config, pretraining_config


def load_config(model_config=default_model_config):
    shared = EasyDict(__name__="All-in-One Configuration")
    shared["output_dir_path"] = ".output"  # output directory
    # merge configs
    shared.update(dataset_config)
    shared.update(tokenizer_config)
    # seperable training
    shared["pretraining"] = pretraining_config
    shared["finetuning"] = sft_config
    # model config
    shared.update(model_config)
    return shared
