# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import torch
from easydict import EasyDict

# exported model configuration
model_config = EasyDict(__name__="Model Configuration - v1")

# OUTPUT CONFIG
model_config.output_path = ".output"

# DATASET CONFIG
model_config.hf_corpus = "facebook/empathetic_dialogues"  # hugging face corpora
model_config.trust_source = True  # MAKE SURE REMOTE SCRIPT ARE SAFE!
model_config.test_ratio = (
    0.2  # train test split ratio (only if dataset require manual split)
)
model_config.eval_ratio = (
    0.5  # test eval split ratio (only if dataset require manual split)
)

# TOKENIZER CONFIG
model_config.tokenizer_strategy = "subword-level"  # word-level or subword-level
model_config.pre_tokenizer_strategy = "byte-level"  # whitespace or byte-level
model_config.vocab_size = 50000  # max vocabulary size
model_config.min_frequency = 2  # minimum number of times a token must appear in the training corpus to be added to vocab
model_config.tokenizer_filename = "tokenizer.json"
model_config.special_tokens = {
    "unknown": "[UNK]",
    "padding": "[PAD]",
    "begin": "[BOS]",
    "end": "[EOS]",
    "separator": "[SEP]",
}  # special tokens
model_config.force_retrain_tokenizer = (
    True  # retrain a tokenizer even the tokenizer already exist
)

# MODEL CONFIG
model_config.random_seed = 42
model_config.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)  # torch device
model_config.num_layers = 4  # num of decoder layer (model depth)
model_config.num_heads = (
    4  # num of attention heads (must be divisor of d_model: d_model // h)
)
model_config.d_model = 256  # model dimension (dim of the vector embedding)
model_config.d_ff = 1024  # dim of the feed forward block (4 * d_model)
model_config.dropout = 0.1  # model overall dropout rate
model_config.estimated_seq_len = (
    200  # estimation for sequence length (model input+output capacity)
)

# TRAINING CONFIG
model_config.eval = True  # enables evaluation during training
model_config.mixed_precision = True  # automatic mixed precision by default
model_config.batch_size = 16  # train batch size
model_config.num_epochs = 30  # train epochs
model_config.lr = 0.0001  # learning rate
model_config.weight_decay = 0.1  # optimizer weight decay
model_config.model_ckpt_path = "mlm_ckpt_epoch_{}.pth"  # model checkpoint path
model_config.model_output_path = "model.pth"  # final model output path
model_config.continous_train = True  # continue latest checkpoint training
