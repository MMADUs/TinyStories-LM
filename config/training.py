# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

from easydict import EasyDict

#######################
# pre-training config #
#######################

pretraining_config = EasyDict(__name__="Pre-training Configuration")

pretraining_config.batch_size = 16  # train batch size
pretraining_config.num_epochs = 3  # train epochs
pretraining_config.lr = 0.0001  # learning rate
pretraining_config.weight_decay = 0.01  # optimizer weight decay
pretraining_config.cross_entropy_ignore_index = (
    -100
)  # ignore index for cross entropy loss
pretraining_config.label_smoothing = (
    0.1  # label smoothing factor for cross entropy loss
)
pretraining_config.lr_warmup_percentage = (
    0.03  # percentage of total training steps for learning rate warmup
)
pretraining_config.clip_grad_max_norm = 1.0  # max norm for gradient clipping

# pretrained checkpoint
pretraining_config.model_ckpt_filename = "pretrained_ckpt_{}.pth"
pretraining_config.callback_metrics_mode = "min"  # min or max, tells the model if the next metric improvement goes to minimum or maximum direction
pretraining_config.ckpt_epsilon = 0  # minimum improvement to save a new checkpoint


###################
# training config #
###################

training_config = EasyDict(__name__="Training Configuration")

training_config.batch_size = 16  # train batch size
training_config.num_epochs = 30  # train epochs
training_config.lr = 0.0001  # learning rate
training_config.weight_decay = 0.01  # optimizer weight decay
training_config.cross_entropy_ignore_index = -100  # ignore index for cross entropy loss
training_config.label_smoothing = 0.1  # label smoothing factor for cross entropy loss
training_config.lr_warmup_percentage = (
    0.03  # percentage of total training steps for learning rate warmup
)
training_config.clip_grad_max_norm = 1.0  # max norm for gradient clipping

# model checkpoint
training_config.model_ckpt_filename = (
    "model_ckpt_{}.pth"  # model checkpoint path with version placeholder
)

training_config.callback_metrics_mode = "min"  # # min or max, tells the model if the next metric improvement goes to minimum or maximum direction
training_config.ckpt_epsilon = 0.01  # minimum improvement to save a new checkpoint
training_config.early_stopping_patience = 5  # early stopping patience (in epochs)
training_config.early_stopping_epsilon = (
    0.001  # minimum improvement to reset early stopping counter
)
