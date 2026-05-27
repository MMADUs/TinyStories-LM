# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

from pathlib import Path

from tokenizers import Tokenizer, Regex
from tokenizers.normalizers import Sequence, Replace, Strip
from tokenizers.models import WordLevel, BPE
from tokenizers.trainers import WordLevelTrainer, BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


def build_normalizer(config):
    """helper function to build tokenizer normalizer"""
    normalizers = []

    if config.get("normalize_whitespace", True):
        # collapse multiple whitespace chars into one normal space
        normalizers.append(Replace(Regex(r"\s+"), " "))

        # remove leading/trailing whitespace
        normalizers.append(Strip())

    if not normalizers:
        return None

    return Sequence(normalizers)


def extract_from_corpus(ds):
    """custom extractor function, depends on the dataset structure"""
    for item in ds:
        text = item.get("text", "")
        if text:  # skip empty items
            yield text


def get_tokenizer(config) -> Tokenizer:
    """get an existing tokenizer from disk

    Args:
        config: model configuration dictionary

    Returns:
        tokenizer: trained tokenizer
    """
    output_dir = Path(config["output_dir_path"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_filename = config["tokenizer_filename"]
    tokenizer_path = output_dir / tokenizer_filename

    if Path(tokenizer_path).exists():
        print("tokenizer found at:", tokenizer_path)
        return Tokenizer.from_file(str(tokenizer_path))
    else:
        raise FileNotFoundError(f"tokenizer not found at {tokenizer_path}")


def get_or_train_tokenizer(config, ds) -> Tokenizer:
    """get existing tokenizer from disk or train a new one from dataset

    Args:
        config: model configuration dictionary
        ds: dataset to train the tokenizer on

    Returns:
        tokenizer: trained tokenizer
    """
    output_dir = Path(config["output_dir_path"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_filename = config["tokenizer_filename"]
    tokenizer_path = output_dir / tokenizer_filename

    force_retrain = config["force_retrain_tokenizer"]

    if Path(tokenizer_path).exists() and not force_retrain:
        print("tokenizer found at:", tokenizer_path)
        return Tokenizer.from_file(str(tokenizer_path))

    print("tokenizer not found, tokenizing corpus...")

    special_tokens_dict = config["special_tokens"]
    special_tokens_list = list(special_tokens_dict.values())

    unk_token = special_tokens_dict["unknown"]
    sep_token = special_tokens_dict["separator"]

    # extract text from corpus
    corpus = extract_from_corpus(ds)

    max_vocab_size = config["vocab_size"]
    strategy = config["tokenizer_strategy"]
    min_freq = config["min_frequency"]

    # tokenizer normalizer
    normalizer = build_normalizer(config)

    pre_tokenizer_strategy = config["pre_tokenizer_strategy"]

    if pre_tokenizer_strategy == "whitespace":
        pre_tokenizer = Whitespace()
    elif pre_tokenizer_strategy == "byte-level":
        pre_tokenizer = ByteLevel()
    else:
        raise ValueError(f"Unknown pre-tokenizer strategy: {pre_tokenizer_strategy}")

    # subword level
    if strategy == "subword-level":
        model = BPE(unk_token=unk_token)
        tokenizer = Tokenizer(model)

        tokenizer.normalizer = normalizer
        tokenizer.pre_tokenizer = pre_tokenizer

        if isinstance(pre_tokenizer, ByteLevel):
            tokenizer.post_processor = ByteLevelProcessor()
            tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=max_vocab_size,
            min_frequency=min_freq,
            special_tokens=special_tokens_list,
        )

        tokenizer.train_from_iterator(corpus, trainer)
    # word level
    elif strategy == "word-level":
        if isinstance(pre_tokenizer, ByteLevel):
            raise ValueError(
                "Byte-level pre-tokenizer is incompatible with WordLevel model."
            )

        model = WordLevel(unk_token=unk_token)
        tokenizer = Tokenizer(model)

        tokenizer.normalizer = normalizer
        tokenizer.pre_tokenizer = pre_tokenizer

        trainer = WordLevelTrainer(
            vocab_size=max_vocab_size,
            min_frequency=min_freq,
            special_tokens=special_tokens_list,
        )

        tokenizer.train_from_iterator(corpus, trainer)
    # error
    else:
        raise ValueError(f"Unknown tokenizer strategy: {strategy}")

    # save tokenizer
    tokenizer.save(str(tokenizer_path))

    print(f"tokenizing finished, tokenizer saved to {tokenizer_path}")
    print(f"final vocab size: {tokenizer.get_vocab_size()}")

    return tokenizer
