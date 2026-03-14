# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import os

# # disable parallelism warning from tokenizers
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import pickle
import json

from config.lm_v1_9M import _9M_config as model_9M_params
from config.lm_v1_19M import _19M_config as model_19M_params
from config.lm_v1_40M import _40M_config as model_40M_params

from config.utils import load_config
from corpus import get_corpus_dataloaders
from instruct import get_sft_dataloaders
from pretrain import pretrain_model
from finetune import finetune_model


def main():
    parser = argparse.ArgumentParser(description="trainer script arg parser")

    parser.add_argument(
        "--model_size", choices=["9M", "19M", "40M"], required=True, help="model size to train"
    )
    parser.add_argument(
        "--version", required=True, help="version name for the training run"
    )
    parser.add_argument(
        "--phase",
        choices=["pretraining", "finetuning"],
        required=True,
        help="training phase",
    )
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

    # select config
    if args.model_size == "9M":
        model_config = model_9M_params
    elif args.model_size == "19M":
        model_config = model_19M_params
    elif args.model_size == "40M":
        model_config = model_40M_params
    else:
        raise ValueError("invalid model size")

    # load config
    config = load_config(model_config)
    print(json.dumps(config, indent=4, default=str))

    # training
    if args.phase == "pretraining":
        train_dl, val_dl, tokenizer = get_corpus_dataloaders(config, is_test=args.test)

        history = pretrain_model(
            config,
            train_dl,
            val_dl,
            tokenizer,
            version=args.version,
            initial_train=(args.state == "initial"),
            load_from_epoch=args.load_epoch,
        )
    elif args.phase == "finetuning":
        train_dl, val_dl, tokenizer = get_sft_dataloaders(config, is_test=args.test)

        history = finetune_model(
            config,
            train_dl,
            val_dl,
            tokenizer,
            version=args.version,
            initial_train=(args.state == "initial"),
            load_from_epoch=args.load_epoch,
        )
    else:
        raise ValueError("invalid training phase")

    with open(f"{config['output_dir_path']}/history_{args.version}.pkl", "wb") as f:
        pickle.dump(history, f)


# python script.py --model_size 40M --version v1_40M --phase (pretraining|finetuning) --state (initial|resume) --load_epoch N --test
if __name__ == "__main__":
    main()
