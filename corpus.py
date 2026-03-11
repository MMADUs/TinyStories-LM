# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

from typing import List, Dict
from datasets import load_dataset

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import get_or_train_tokenizer
from utils import DeviceDataLoader


class TinyStoriesCorpus(Dataset):
    """
    Custom torch dataset class for TinyStories data. It takes the raw dataset and a trained tokenizer.

    Args:
    - config: model configuration dictionary
    - data: raw dataset to be processed
    - tokenizer: trained tokenizer to tokenize the text data
    """

    def __init__(self, config, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

        # special token IDs
        special_tokens_dict = config["special_tokens"]
        self.bos_id = tokenizer.token_to_id(special_tokens_dict["begin"])
        self.eos_id = tokenizer.token_to_id(special_tokens_dict["end"])

        # max sequence length (truncate to prevent OOM)
        self.max_seq_len = config["max_seq_truncation"]

        # ignore index
        self.ignore_index = config["pretraining"]["cross_entropy_ignore_index"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get("text", "")  # empty if context is not available

        # tokenize
        text_ids = self.tokenizer.encode(text).ids

        # concatenate full sequence: [BOS] text [EOS]
        tokens = [self.bos_id] + text_ids + [self.eos_id]

        # truncate to max_seq_len to slice long seq
        if len(tokens) > self.max_seq_len:
            # slice and add EOS
            tokens = tokens[: self.max_seq_len - 1] + [self.eos_id]

        input_ids = torch.tensor(tokens, dtype=torch.long)

        return {
            "input_ids": input_ids,
        }


def create_collate_fn(config, tokenizer):
    """
    Function generator for the custom collate_fn dataloader.

    Args:
    - config: model configuration dictionary
    - tokenizer: trained tokenizer to get special token IDs
    """
    # get padding ID
    padding = config["special_tokens"]["padding"]
    pad_id = tokenizer.token_to_id(padding)

    # custom collate_fn to be passed to dataloader
    def _collate_fn(batch: List[Dict]):
        """
        Custom collate_fn to pad variable length sequences in the batch to the same length.
        """
        # get data from batch
        input_ids = [item["input_ids"] for item in batch]

        # find max length in batch
        max_len = max([x.size(0) for x in input_ids])

        # pad sequences to max_len
        def pad_tensor(tensor, pad_value):
            return torch.cat(
                [
                    tensor,
                    torch.full(
                        (max_len - tensor.size(0),), pad_value, dtype=tensor.dtype
                    ),
                ]
            )

        # we need to make all data has a fixed tensor size
        # so we expand tensor to max_len, fill the empty value with pad_value
        input_ids = torch.stack([pad_tensor(t, pad_value=pad_id) for t in input_ids])

        # dataloader output
        return {
            "input_ids": input_ids,
        }

    return _collate_fn


def get_corpus_dataloaders(config):
    """
    Returns train and val corpus dataloaders.

    Args:
    - config: model configuration dictionary
    """
    trust = config["trust_source"]

    # load hf corpora
    ds_train = load_dataset(config["hf_corpus"], split="train", trust_remote_code=trust)
    ds_val = load_dataset(
        config["hf_corpus"], split="validation", trust_remote_code=trust
    )

    # get tokenizer
    tokenizer = get_or_train_tokenizer(config, ds_train)

    # build dataset
    train_ds = TinyStoriesCorpus(config, ds_train, tokenizer)
    val_ds = TinyStoriesCorpus(config, ds_val, tokenizer)

    batch_size = config["pretraining"]["batch_size"]
    val_test_batch_size = max(1, batch_size // 2)

    # create collate_fn
    collate_fn = create_collate_fn(config, tokenizer)

    num_workers = config["num_workers"]

    # dataloader
    train_dl = DataLoader(
        train_ds,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=val_test_batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    device = config["device"]

    return (
        DeviceDataLoader(train_dl, device),
        DeviceDataLoader(val_dl, device),
        tokenizer,
    )
