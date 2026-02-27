# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import torch
from easydict import EasyDict

# exported model configuration
model_config = EasyDict(__name__="Model Configuration v1")

# 1. OUTPUT DIR
model_config.output_dir_path = ".output"

# 2. DATASET CONFIG
model_config.hf_corpus = "facebook/empathetic_dialogues"  # hugging face corpora
model_config.trust_source = True  # MAKE SURE REMOTE SCRIPT ARE SAFE!
model_config.test_ratio = (
    0.2  # train test split ratio (only if dataset require manual split)
)
model_config.eval_ratio = (
    0.5  # test eval split ratio (only if dataset require manual split)
)

# 3. TOKENIZER CONFIG
model_config.tokenizer_strategy = "subword-level"  # word-level or subword-level
model_config.pre_tokenizer_strategy = "byte-level"  # whitespace or byte-level
model_config.vocab_size = 30000  # max vocabulary size
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
    False  # retrain a tokenizer even the tokenizer already exist
)

# 4. MODEL CONFIG
model_config.random_seed = 42
model_config.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)  # torch device
model_config.init_std = 0.02  # parameter initialization standard deviation
model_config.num_layers = 4  # num of decoder layer (model depth)
model_config.num_heads = (
    4  # num of attention heads (must be divisor of d_model: d_model // h)
)
model_config.d_model = 256  # model dimension (dim of the vector embedding)
model_config.d_ff = 1024  # dim of the feed forward block (4 * d_model)
model_config.dropout = 0.1  # model dropout rate
model_config.norm_strategy = "RMSNorm"  # normalization strategy (RMSNorm or LayerNorm)
model_config.max_seq_truncation = 200 # maximum input sequence, anything longer will be truncanated and immediately ended with EOS
model_config.estimated_seq_len = 500  # estimation for sequence length (max model input+output capacity during inference)

# 5. TRAINING CONFIG
# training loop
model_config.num_workers = 0 # dataloader num workers
model_config.batch_size = 16  # train batch size
model_config.num_epochs = 30  # train epochs
model_config.lr = 0.0001  # learning rate
model_config.weight_decay = 0.01  # optimizer weight decay
model_config.cross_entropy_ignore_index = -100  # ignore index for cross entropy loss
model_config.label_smoothing = 0.1  # label smoothing factor for cross entropy loss
model_config.lr_warmup_percentage = (
    0.03  # percentage of total training steps for learning rate warmup
)
model_config.clip_grad_max_norm = 1.0  # max norm for gradient clipping
model_config.callback_metrics_mode = "min"  # # min or max, tells the model if the next metric improvement goes to minimum or maximum direction
# model checkpoint
model_config.model_ckpt_filename = (
    "model_ckpt_{}.pth"  # model checkpoint path with version placeholder
)
model_config.ckpt_epsilon = 0.01  # minimum improvement to save a new checkpoint
# training early stopping
model_config.early_stopping_patience = 5  # early stopping patience (in epochs)
model_config.early_stopping_epsilon = (
    0.001  # minimum improvement to reset early stopping counter
)
# learning rate scheduler on plateau
model_config.lr_reduction_factor = 0.5  # factor to reduce learning rate on plateau
model_config.lr_reduction_patience = (
    3  # number of epochs with no improvement to wait before reducing learning rate
)
model_config.lr_reduction_cooldown = 1  # number of epochs to wait after reducing learning rate before resuming normal operation
model_config.lr_reduction_epsilon = (
    0.001  # minimum improvement to reset learning rate reduction counter
)
model_config.lr_reduction_min_lr = 1e-6  # minimum learning rate after reduction
