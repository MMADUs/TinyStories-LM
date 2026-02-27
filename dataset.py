# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

from typing import List, Dict
from datasets import load_dataset

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import get_or_train_tokenizer


class DialogueDataset(Dataset):
    """
    Custom torch dataset class for dialogue data. It takes the raw dataset and a trained tokenizer.

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
        self.sep_id = tokenizer.token_to_id(special_tokens_dict["separator"])
        self.pad_id = tokenizer.token_to_id(special_tokens_dict["padding"])

        # max sequence length (truncate to prevent OOM)
        self.max_seq_len = config["max_seq_truncation"]

        # ignore index
        self.ignore_index = config["cross_entropy_ignore_index"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        context = sample.get("context", "")  # empty if context is not available
        prompt = sample["prompt"]
        response = sample["utterance"]

        # tokenize each section
        context_ids = self.tokenizer.encode(context).ids if context else []
        prompt_ids = self.tokenizer.encode(prompt).ids
        response_ids = self.tokenizer.encode(response).ids

        # concatenate full sequence: [BOS] context [SEP] prompt [SEP] response [EOS]
        tokens = (
            [self.bos_id]
            + context_ids
            + [self.sep_id]
            + prompt_ids
            + [self.sep_id]
            + response_ids
            + [self.eos_id]
        )

        # truncate to max_seq_len to prevent OOM on long sequences
        if len(tokens) > self.max_seq_len:
            tokens = tokens[: self.max_seq_len - 1] + [self.eos_id]

        input_ids = torch.tensor(tokens, dtype=torch.long)

        labels = input_ids.clone()

        # get index position of separator token
        sep_positions = (input_ids == self.sep_id).nonzero(as_tuple=True)[0]

        if len(sep_positions) > 1:
            cut = sep_positions[-1].item()  # last SEP separates prompt from response
        else:
            cut = 0  # fallback

        # cross entropy loss ignores index with value -100
        # this way we use to mask: [BOS] context [SEP] prompt [SEP]
        # remaining for label: response [EOS]
        labels[: cut + 1] = self.ignore_index

        # padding mask: 1/Yes for all tokens (since we dont have padding)
        padding_mask = torch.ones_like(input_ids, dtype=torch.bool)

        # pass to collate_fn
        return {
            "input_ids": input_ids,
            "labels": labels,
            "padding_mask": padding_mask,
        }


def create_collate_fn(config, tokenizer):
    """
    Fuction generator for the custom collate_fn dataloader.

    Args:
    - config: model configuration dictionary
    - tokenizer: trained tokenizer to get special token IDs
    """
    # get padding ID
    padding = config["special_tokens"]["padding"]
    pad_id = tokenizer.token_to_id(padding)

    # get ignore index
    ignore_index = config["cross_entropy_ignore_index"]

    # custom collate_fn to be passed to dataloader
    def _collate_fn(batch: List[Dict]):
        """
        Custom collate_fn to pad variable length sequences in the batch to the same length.
        """
        # get data from batch
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]
        padding_mask = [item["padding_mask"] for item in batch]

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

        # fill label padding with ignore_index, so cross entropy loss will ignore those positions
        labels = torch.stack([pad_tensor(t, pad_value=ignore_index) for t in labels])

        # mask padding value is 0/False, which means the model should not attend to those positions
        padding_mask = torch.stack(
            [pad_tensor(t, pad_value=False) for t in padding_mask]
        )

        # dataloader output
        return {
            "input_ids": input_ids,
            "labels": labels,
            "padding_mask": padding_mask,
        }

    return _collate_fn


class DeviceDataLoader:
    """
    DeviceDataLoader is a wrapper around torch DataLoader that moves data to the specified device.

    Args:
    - dl: torch DataLoader to wrap
    - device: torch device to move data to
    """

    def __init__(self, dl: DataLoader, device):
        self.dl = dl
        self.device = device

    def _to_device(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        else:
            return batch  # leave other types untouched

    def __iter__(self):
        for batch in self.dl:
            yield self._to_device(batch)

    def __len__(self):
        return len(self.dl)


def get_dataloaders(config):
    """
    Returns train, val, and test dataloaders from the given config.

    Args:
    - config: model configuration dictionary
    """
    trust = config["trust_source"]

    # load hf corpora
    ds_train = load_dataset(config["hf_corpus"], split="train", trust_remote_code=trust)
    ds_val = load_dataset(
        config["hf_corpus"], split="validation", trust_remote_code=trust
    )
    ds_test = load_dataset(config["hf_corpus"], split="test", trust_remote_code=trust)

    # get tokenizer
    tokenizer = get_or_train_tokenizer(config, ds_train)

    # build dataset
    train_ds = DialogueDataset(config, ds_train, tokenizer)
    val_ds = DialogueDataset(config, ds_val, tokenizer)
    test_ds = DialogueDataset(config, ds_test, tokenizer)

    batch_size = config["batch_size"]
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
    test_dl = DataLoader(
        test_ds,
        batch_size=val_test_batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    device = config["device"]

    # max input sequence length
    lengths = [len(sample["input_ids"]) for sample in train_ds]

    max_len = max(lengths)
    min_len = min(lengths)
    avg_len = sum(lengths) / len(lengths)

    print(f"max input seq length: {max_len}")
    print(f"min input seq length: {min_len}")
    print(f"avg seq length: {avg_len:.2f}")

    return (
        DeviceDataLoader(train_dl, device),
        DeviceDataLoader(val_dl, device),
        DeviceDataLoader(test_dl, device),
        tokenizer,
    )
