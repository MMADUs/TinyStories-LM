# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

from easydict import EasyDict

#######################
# pre-training config #
#######################

pretraining_config = EasyDict(__name__="Pre-training Configuration")

# most critical parameter during training
# maximum input token sequence, anything longer will be truncanated and immediately ended with EOS
# number is taken from the avg of the tokenized seq len from the entire corpus
pretraining_config.max_seq_truncation = 512

pretraining_config.batch_size = 8  # train batch size

# num epochs NEEDS to be adjusted when resuming training from a checkpoint.
pretraining_config.num_epochs = 2

pretraining_config.lr = 0.0001  # learning rate (common default: 1e-4)
pretraining_config.weight_decay = 0.01  # optimizer weight decay

pretraining_config.cross_entropy_ignore_index = (
    -100
)  # ignore index for cross entropy loss
pretraining_config.label_smoothing = (
    0.0  # label smoothing factor for cross entropy loss
)

pretraining_config.lr_warmup_percentage = (
    0.03  # percentage of total training steps for learning rate warmup
)
pretraining_config.clip_grad_max_norm = 1.0  # max norm for gradient clipping

# pretrained checkpoint
pretraining_config.model_ckpt_filename = "pretrained_ckpt_{}"
pretraining_config.ckpt_format = ".pth"
pretraining_config.callback_metrics_mode = "min"  # min or max, tells the model if the next metric improvement goes to minimum or maximum direction
pretraining_config.ckpt_epsilon = 0  # minimum improvement to save a new checkpoint

# append once to history every N steps
pretraining_config.append_train_history_step = 1000
pretraining_config.append_val_history_step = 100


#################################
# supervised fine-tuning config #
#################################

sft_config = EasyDict(__name__="Supervised Fine-Tuning Configuration")

# most critical parameter during training
# maximum input token sequence, anything longer will be truncanated and immediately ended with EOS
# number is taken from the avg of the tokenized seq len from the entire corpus
sft_config.max_seq_truncation = 512

sft_config.batch_size = 8  # train batch size

# num epochs NEEDS to be adjusted when resuming training from a checkpoint.
sft_config.num_epochs = 1  # 1 epoch = 1 runs

sft_config.lr = 0.00002  # learning rate (common default: 2e-5)
sft_config.weight_decay = 0.01  # optimizer weight decay

sft_config.cross_entropy_ignore_index = -100  # ignore index for cross entropy loss
sft_config.label_smoothing = 0.05  # label smoothing factor for cross entropy loss

sft_config.lr_warmup_percentage = (
    0.03  # percentage of total training steps for learning rate warmup
)
sft_config.clip_grad_max_norm = 1.0  # max norm for gradient clipping

# model checkpoint
sft_config.model_ckpt_filename = "model_ckpt_{}"
sft_config.ckpt_format = ".pth"
sft_config.callback_metrics_mode = "min"  # # min or max, tells the model if the next metric improvement goes to minimum or maximum direction
sft_config.ckpt_epsilon = 0.01  # minimum improvement to save a new checkpoint

# append once to history every N steps
sft_config.append_train_history_step = 1000
sft_config.append_val_history_step = 100


#####################################
# LoRA (low-rank adaptation) config #
#####################################

lora_config = EasyDict(__name__="LoRA (Low-Rank Adaptation) Configuration")

lora_config.target_modules = (
    "w_q",
    "w_v",
)  # target linear layers to replace with LoRA adapters

# rank = bottleneck dimension of the LoRA adapter
# LoRA learns: original_output + B(Ax)
# A projects from d_model -> rank
# B projects from rank -> d_model
# smaller rank = fewer trainable params, less capacity
# larger rank = more trainable params, more adaptation capacity
lora_config.rank = 8

lora_config.alpha = 16  # LoRA scaling factor, common rule = 2 * rank
lora_config.dropout = (
    0.05  # LoRA dropout rate, dropout applied only on the LoRA branch input
)
lora_config.is_quantized = (
    True  # QLoRA freeze base layer and store/use it in 4-bit quantized form
)
