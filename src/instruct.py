# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import json
from pathlib import Path
from typing import List, Dict

from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader

from src.tokenizer import get_tokenizer
from src.utils import DeviceDataLoader


def make_structured_data(data):
    """
    merge row-wise data into structured data with fields: features, words, summary, story.
    """
    cleaned_data = []
    current_story = {"features": "", "words": "", "summary": "", "story": []}

    for line in data:
        line = line.strip()

        # features: conditioning output to express story
        if line.startswith("Features:"):
            current_story["features"] = line[len("Features: ") :]
        # words: conditioning output to contain specified words
        elif line.startswith("Words:"):
            current_story["words"] = line[len("Words: ") :]
        # summary: the story prompt
        elif line.startswith("Summary:"):
            current_story["summary"] = line[len("Summary: ") :]
        # story: generated story based on prompt and conditioning
        elif line == "Story:":
            current_story["story"] = []
        # end of row
        elif line == "<|endoftext|>":
            current_story["story"] = " ".join(current_story["story"]).strip()
            cleaned_data.append(current_story)
            # reset for next data
            current_story = {"features": "", "words": "", "summary": "", "story": []}
        else:
            current_story["story"].append(line)

    return cleaned_data


def save_structured_data(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


def load_structured_data(path):
    with open(path, "r") as f:
        return json.load(f)


class TinyStoriesSFT(Dataset):
    """supervised fine tuning torch dataset

    Args:
        config: model configuration dictionary
        data: raw dataset to be processed
        tokenizer: trained tokenizer to tokenize the text data
    """

    def __init__(self, config, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

        # special token IDs
        special_tokens_dict = config["special_tokens"]

        self.bos_id = tokenizer.token_to_id(special_tokens_dict["begin"])
        self.eos_id = tokenizer.token_to_id(special_tokens_dict["end"])

        # conditioning special tokens
        self.features_start = tokenizer.token_to_id(
            special_tokens_dict["features_start"]
        )
        self.features_end = tokenizer.token_to_id(special_tokens_dict["features_end"])
        self.words_start = tokenizer.token_to_id(special_tokens_dict["words_start"])
        self.words_end = tokenizer.token_to_id(special_tokens_dict["words_end"])
        self.summary_start = tokenizer.token_to_id(special_tokens_dict["summary_start"])
        self.summary_end = tokenizer.token_to_id(special_tokens_dict["summary_end"])
        self.story_start = tokenizer.token_to_id(special_tokens_dict["story_start"])
        self.story_end = tokenizer.token_to_id(special_tokens_dict["story_end"])

        # max sequence length for truncation
        self.max_seq_len = config["finetuning"]["max_seq_truncation"]

        # ignore index
        self.ignore_index = config["finetuning"]["cross_entropy_ignore_index"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        features = item.get("features", "")
        words = item.get("words", "")
        summary = item.get("summary", "")
        story = item.get("story", "")

        # tokenize
        features_ids = self.tokenizer.encode(features).ids
        words_ids = self.tokenizer.encode(words).ids
        summary_ids = self.tokenizer.encode(summary).ids
        story_ids = self.tokenizer.encode(story).ids

        # concatenate full input sequence
        tokens = (
            [self.bos_id]  # [BOS]
            + [self.features_start]  # <features>
            # features ctx
            + features_ids
            + [self.features_end]  # </features>
            + [self.words_start]  # <words>
            # words ctx
            + words_ids
            + [self.words_end]  # </words>
            + [self.summary_start]  # <summary>
            # summary ctx
            + summary_ids
            + [self.summary_end]  # </summary>
            + [self.story_start]  # <story>
            # story to be predicted
            + story_ids
            + [self.story_end]  # </story>
            + [self.eos_id]
        )

        # mask BOS until <story>
        # prediction begins after <story>
        prompt_tokens = (
            [self.bos_id]
            + [self.features_start]
            + features_ids
            + [self.features_end]
            + [self.words_start]
            + words_ids
            + [self.words_end]
            + [self.summary_start]
            + summary_ids
            + [self.summary_end]
            + [self.story_start]
        )

        labels = (
            [self.ignore_index] * len(prompt_tokens)  # masked
            + story_ids  # predict
            + [self.story_end]  # predict
            + [self.eos_id]  # predict
        )

        # truncate to max_seq_len to slice long seq
        if len(tokens) > self.max_seq_len:
            # slice and add EOS
            tokens = tokens[: self.max_seq_len - 1] + [self.eos_id]
            labels = labels[: self.max_seq_len - 1] + [self.eos_id]

        assert len(tokens) == len(
            labels
        ), f"tokens {len(tokens)} != labels {len(labels)}"

        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class CollateFn:
    def __init__(self, pad_id, ignore_index):
        self.pad_id = pad_id
        self.ignore_index = ignore_index

    def __call__(self, batch: List[Dict]):
        # get data from batch
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]

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
        input_ids = torch.stack(
            [pad_tensor(t, pad_value=self.pad_id) for t in input_ids]
        )
        labels = torch.stack(
            [pad_tensor(t, pad_value=self.ignore_index) for t in labels]
        )

        # dataloader output
        return {
            "input_ids": input_ids,
            "labels": labels,
        }


def get_sft_dataloaders(config, is_test=False):
    """returns train and val SFT dataloaders

    Args:
        config: model configuration dictionary
        is_test: return few samples of the dataset for testing

    Returns:
        train_dl: training dataloader
        val_dl: validation dataloader
        tokenizer: trained tokenizer
    """
    trust = config["trust_source"]

    # load hf corpora
    ds_train = load_dataset(
        config["hf_instruct"], split="train", trust_remote_code=trust
    )
    ds_val = load_dataset(
        config["hf_instruct"], split="validation", trust_remote_code=trust
    )

    # finetuning dataset path
    dataset_dir = Path(config["output_dir_path"]) / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_ds_dir = dataset_dir / "sft_train.json"
    val_ds_dir = dataset_dir / "sft_val.json"

    # preprocessing if not exists, else load from disk
    if train_ds_dir.exists() and val_ds_dir.exists():
        print("loading structured data from disk...")

        ds_train = load_structured_data(train_ds_dir)
        ds_val = load_structured_data(val_ds_dir)
    else:
        print("running text preprocessing...")

        ds_train = make_structured_data(ds_train["text"])
        ds_val = make_structured_data(ds_val["text"])

        save_structured_data(ds_train, train_ds_dir)
        save_structured_data(ds_val, val_ds_dir)

    if is_test:
        # for testing, we use a smaller subset of the data to speed up the process
        ds_train = ds_train[:2500]
        ds_val = ds_val[:2500]

    # get tokenizer
    tokenizer = get_tokenizer(config)

    # build dataset
    train_ds = TinyStoriesSFT(config, ds_train, tokenizer)
    val_ds = TinyStoriesSFT(config, ds_val, tokenizer)

    batch_size = config["finetuning"]["batch_size"]
    val_test_batch_size = max(1, batch_size // 2)

    padding = config["special_tokens"]["padding"]
    pad_id = tokenizer.token_to_id(padding)

    ignore_index = config["finetuning"]["cross_entropy_ignore_index"]

    # create collate_fn
    collate_fn = CollateFn(pad_id, ignore_index)

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
