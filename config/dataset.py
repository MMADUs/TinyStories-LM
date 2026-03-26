# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

from easydict import EasyDict

##################
# dataset config #
##################

dataset_config = EasyDict(__name__="Dataset Configuration")

dataset_config.hf_corpus = "roneneldan/TinyStories"  # pretraining corpus
dataset_config.hf_instruct = "roneneldan/TinyStoriesInstruct"  # supervised dataset
dataset_config.trust_source = True  # MAKE SURE REMOTE SCRIPT ARE SAFE!

dataset_config.test_ratio = (
    0.2  # train test split ratio (only if dataset require manual split)
)
dataset_config.eval_ratio = (
    0.5  # test eval split ratio (only if dataset require manual split)
)

dataset_config.num_workers = 0  # dataloader num workers


####################
# tokenizer config #
####################

tokenizer_config = EasyDict(__name__="Tokenizer Configuration")

tokenizer_config.tokenizer_strategy = "subword-level"  # word-level or subword-level
tokenizer_config.pre_tokenizer_strategy = "byte-level"  # whitespace or byte-level
tokenizer_config.vocab_size = 20000  # max vocabulary size
tokenizer_config.min_frequency = 2  # minimum number of times a token must appear in the training corpus to be added to vocab
tokenizer_config.tokenizer_filename = "tokenizer.json"
tokenizer_config.special_tokens = {
    "unknown": "[UNK]",
    "padding": "[PAD]",
    "begin": "[BOS]",
    "end": "[EOS]",
    "separator": "[SEP]",
    # sft tokens
    "features_start": "<features>",
    "features_end": "</features>",
    "words_start": "<words>",
    "words_end": "</words>",
    "summary_start": "<summary>",
    "summary_end": "</summary>",
    "story_start": "<story>",
    "story_end": "</story>",
}  # special tokens
tokenizer_config.force_retrain_tokenizer = (
    False  # retrain a tokenizer even the tokenizer already exist
)
