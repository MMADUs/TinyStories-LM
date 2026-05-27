# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import os

# # disable parallelism warning from tokenizers
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import pickle
import json

from config.std_9M import std_9M
from config.std_21M import std_21M
from config.std_43M import std_43M

from config.MoE_9M import MoE_9M
from config.MoE_21M import MoE_21M
from config.MoE_43M import MoE_43M

from config.utils import load_config
from src.corpus import get_corpus_dataloaders
from src.instruct import get_sft_dataloaders
from src.train import train_model

STD_MODELS = {
    "9M": std_9M,
    "21M": std_21M,
    "43M": std_43M,
}

MOE_MODELS = {
    "9M": MoE_9M,
    "21M": MoE_21M,
    "43M": MoE_43M,
}


def main():
    parser = argparse.ArgumentParser(description="trainer script arg parser")

    parser.add_argument(
        "--model_size",
        choices=["9M", "21M", "43M"],
        required=True,
        help="model size to train",
    )
    parser.add_argument("--moe", action="store_true", help="using MoE in the model")
    parser.add_argument(
        "--version", required=True, help="version name for the training run"
    )
    parser.add_argument(
        "--phase",
        choices=["pretraining", "finetuning"],
        required=True,
        help="training phase",
    )
    parser.add_argument("--lora", action="store_true", help="enable LoRA finetuning")
    parser.add_argument(
        "--state", choices=["initial", "resume"], required=True, help="training state"
    )
    parser.add_argument(
        "--load_epoch", type=int, default=None, help="epoch to load checkpoint from"
    )
    parser.add_argument("--test", action="store_true", help="run in test mode")

    args = parser.parse_args()

    if args.state == "resume" and args.load_epoch is None:
        parser.error("--load_epoch is required when --state is resume")

    if args.phase == "pretraining" and args.lora:
        parser.error("LoRA should only be used for fine tuning")

    # select config
    try:
        if args.moe:
            model_config = MOE_MODELS[args.model_size]
        else:
            model_config = STD_MODELS[args.model_size]
    except KeyError:
        raise ValueError(f"invalid model size: {args.model_size}")

    # load config
    config = load_config(model_config)
    print(json.dumps(config, indent=4, default=str))

    # training
    if args.phase == "pretraining":
        train_dl, val_dl, tokenizer = get_corpus_dataloaders(config, is_test=args.test)
    elif args.phase == "finetuning":
        train_dl, val_dl, tokenizer = get_sft_dataloaders(config, is_test=args.test)
    else:
        raise ValueError("invalid training phase")

    history = train_model(
        config,
        train_dl,
        val_dl,
        tokenizer,
        stage=args.phase,
        enable_lora=args.lora,
        version=args.version,
        initial_train=(args.state == "initial"),
        load_from_epoch=args.load_epoch,
    )

    with open(
        f"{config['output_dir_path']}/{args.state}_history_{args.version}.pkl", "wb"
    ) as f:
        pickle.dump(history, f)


# pretraining script:
# python script.py --model_size 40M --version v1_40M --phase pretraining --state (initial|resume) --load_epoch N --test
#
# finetuning script:
# python script.py --model_size 40M --version v1_40M --phase finetuning --state (initial|resume) --load_epoch N --test
if __name__ == "__main__":
    main()
