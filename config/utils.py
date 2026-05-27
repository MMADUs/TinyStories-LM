# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

from easydict import EasyDict

from config.dataset import dataset_config, tokenizer_config
from config.training import pretraining_config, sft_config, lora_config


def load_config(model_config):
    shared = EasyDict(__name__="All-in-One Configuration")
    shared["output_dir_path"] = ".output"  # output directory
    # merge configs
    shared.update(dataset_config)
    shared.update(tokenizer_config)
    # seperable training
    shared["pretraining"] = pretraining_config
    shared["finetuning"] = sft_config
    shared["finetuning"]["lora"] = lora_config
    # model config
    shared.update(model_config)
    return shared
