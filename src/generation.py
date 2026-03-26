# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

from pathlib import Path
from typing import Literal, List

import torch

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from src.model import build_model, get_attn_mask
from src.serialization import get_checkpoint_path


class ModelLoader:
    """
    Class for loading model checkpoints.

    Args:
    - config: model configuration dictionary
    - stage: training stage, either "pretraining" or "finetuning"
    - version: version string to identify the model checkpoint to load
    - context: initial dialogue context to prime the model with (optional)
    """

    def __init__(
        self,
        config,
        stage: Literal["pretraining", "finetuning"],
        version,
        load_from_epoch,
    ):
        self.device = config["device"]
        self.config = config
        self.stage_config = config[stage]

        output_dir = Path(config["output_dir_path"])

        # load tokenizer from disk
        tokenizer_filename = config["tokenizer_filename"]
        tokenizer_path = output_dir / tokenizer_filename

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))

        # TODO: this is temporary, should be removed soon when tokenizer is retrained
        self.tokenizer.decoder = ByteLevelDecoder()

        ckpt_path = get_checkpoint_path(
            self.config, self.stage_config, version, load_from_epoch
        )

        print("loading model from:", ckpt_path)

        checkpoint = torch.load(ckpt_path)
        model_state = checkpoint["model"]

        self.model = build_model(self.config, self.tokenizer)
        self.model.load_state_dict(model_state)
        self.model = self.model.to(self.device)
        self.model.eval()

    @property
    def get_configs(self):
        return self.config, self.stage_config

    @property
    def get_tokenizer(self):
        return self.tokenizer

    def autoregressive_attribute(
        self,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    def generate(self, input_ids, special_tokens):
        return self.model.generate(
            input_ids=input_ids,
            special_tokens=special_tokens,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )


def next_word_prediction(
    model_loader: ModelLoader,
    text_input: str = "",
) -> str:
    config, _stage_config = model_loader.get_configs
    tokenizer = model_loader.get_tokenizer

    # load special token ids
    special_tokens_dict = config["special_tokens"]

    bos_id = tokenizer.token_to_id(special_tokens_dict["begin"])
    eos_id = tokenizer.token_to_id(special_tokens_dict["end"])
    pad_id = tokenizer.token_to_id(special_tokens_dict["padding"])

    ctx_ids = tokenizer.encode(text_input).ids if text_input is not None else []

    # initalize input_ids with BOS + init_context
    input_ids = [bos_id] + ctx_ids

    input_ids = torch.tensor(
        input_ids, dtype=torch.long, device=config["device"]
    ).unsqueeze(0)

    special_tokens_ids = {
        "end": eos_id,
        "padding": pad_id,
    }

    output = model_loader.generate(input_ids, special_tokens_ids)

    # decode
    list_ids = output.squeeze(0).tolist()

    cleaned_ids = []

    for token_id in list_ids:
        if token_id in (bos_id, pad_id):
            continue
        if token_id == eos_id:
            break
        cleaned_ids.append(token_id)

    generated_text = tokenizer.decode(cleaned_ids)

    cleaned_text = (
        generated_text.replace(" ,", ",")
        .replace(" .", ".")
        .replace(" !", "!")
        .replace(" ?", "?")
        .replace(" '", "'")
        .replace(" \n", "\n")
        .replace("Ġ", " ")
        .replace("Ċ", "\n")
        .strip()
    )

    return cleaned_text


def generate_story(
    model_loader: ModelLoader, features: List[str], words: List[str], summary: str
):
    config, _stage_config = model_loader.get_configs
    tokenizer = model_loader.get_tokenizer

    special_tokens_dict = config["special_tokens"]

    bos_id = tokenizer.token_to_id(special_tokens_dict["begin"])
    eos_id = tokenizer.token_to_id(special_tokens_dict["end"])
    pad_id = tokenizer.token_to_id(special_tokens_dict["padding"])

    features_start = tokenizer.token_to_id(special_tokens_dict["features_start"])
    features_end = tokenizer.token_to_id(special_tokens_dict["features_end"])
    words_start = tokenizer.token_to_id(special_tokens_dict["words_start"])
    words_end = tokenizer.token_to_id(special_tokens_dict["words_end"])
    summary_start = tokenizer.token_to_id(special_tokens_dict["summary_start"])
    summary_end = tokenizer.token_to_id(special_tokens_dict["summary_end"])
    story_start = tokenizer.token_to_id(special_tokens_dict["story_start"])
    story_end = tokenizer.token_to_id(special_tokens_dict["story_end"])

    # 1. merge lists to string before encoding
    features_str = ", ".join(features) if features else ""
    words_str = ", ".join(words) if words else ""

    features_ids = tokenizer.encode(features_str).ids if features_str else []
    words_ids = tokenizer.encode(words_str).ids if words_str else []
    summary_ids = tokenizer.encode(summary).ids if summary else []

    prompt_tokens = (
        [bos_id]
        + [features_start]
        + features_ids
        + [features_end]
        + [words_start]
        + words_ids
        + [words_end]
        + [summary_start]
        + summary_ids
        + [summary_end]
        + [story_start]
    )

    input_ids = torch.tensor(
        prompt_tokens, dtype=torch.long, device=config["device"]
    ).unsqueeze(0).clone()

    special_tokens_ids = {
        "end": eos_id,
        "padding": pad_id,
    }

    output = model_loader.generate(input_ids, special_tokens_ids)

    list_ids = output.squeeze(0).tolist()

    # skip tokens before and including story_start
    skip_ids = {
        bos_id,
        pad_id,
        features_start,
        features_end,
        words_start,
        words_end,
        summary_start,
        summary_end,
        story_start,
    }

    prompt_len = input_ids.size(1)
    generated_ids = list_ids[prompt_len:] # take the response part only, skip the prompt

    cleaned_ids = []
    for token_id in generated_ids:
        if token_id in skip_ids:
            continue
        if token_id in (eos_id, story_end):  # stop at either
            break
        cleaned_ids.append(token_id)

    story = tokenizer.decode(cleaned_ids)
    story = (
        story.replace(" ,", ",")
        .replace(" .", ".")
        .replace(" !", "!")
        .replace(" ?", "?")
        .replace(" '", "'")
        .replace(" \n", "\n")
        .replace("Ġ", " ")
        .replace("Ċ", "\n")
        .strip()
    )

    return features_str, words_str, summary, story
