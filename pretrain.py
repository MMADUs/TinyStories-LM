# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import time
from typing import Optional
import numpy as np
import torch
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup

from model import build_model, get_attn_mask
from callbacks import get_default_callbacks, TrainingCallback
from utils import time_formatter, set_random_seed


def pretrain_model(
    config,
    train_dl,
    val_dl,
    tokenizer,
    callbacks: Optional[TrainingCallback] = None,
    version: str = "NA",
) -> dict:
    """
    The pretraining loop.

    Args:
    - config: model configuration dictionary
    - train_dl: training dataloader
    - val_dl: validation dataloader
    - tokenizer: trained tokenizer
    - callbacks: TrainingCallback instance for handling training callbacks
    - version: version string to identify the checkpoint file (default: "NA")
    """
    # set random seed for reproducibility
    set_random_seed(config["random_seed"])

    pretrain_config = config["pretraining"]
    device = config["device"]
    init_epoch = 0

    pad_id = tokenizer.token_to_id(config["special_tokens"]["padding"])

    model = build_model(config, tokenizer)
    model = model.to(device)

    lr = pretrain_config["lr"]
    weight_decay = pretrain_config["weight_decay"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # warmup + linear decay learning rate scheduler
    lr_warmup_percentage = pretrain_config["lr_warmup_percentage"]
    remaining_epochs = pretrain_config["num_epochs"] - init_epoch
    num_training_steps = len(train_dl) * remaining_epochs
    num_warmup_steps = int(lr_warmup_percentage * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # criterion
    label_smoothing = pretrain_config["label_smoothing"]
    cross_entropy_ignore_index = pretrain_config["cross_entropy_ignore_index"]

    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=cross_entropy_ignore_index,  # ignore index for padding
        label_smoothing=label_smoothing,  # label smoothing factor
    )

    # gradient scaler
    scaler = GradScaler()

    # init callbacks
    if callbacks is None:
        callbacks = get_default_callbacks(config, pretrain_config, version)

    callbacks.reset()

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_perplexity": [],
        "val_perplexity": [],
    }

    start_time = time.time()

    for epoch in range(init_epoch, pretrain_config["num_epochs"]):
        epoch_start = time.time()
        torch.cuda.empty_cache()

        model.train()
        train_loss = 0.0

        batch_iter = tqdm(train_dl, desc=f"epoch {epoch+1}")

        for batch in batch_iter:
            # input
            input_ids = batch["input_ids"].to(device)
            attention_mask = get_attn_mask(input_ids, pad_id)

            # reset optimizer gradient
            optimizer.zero_grad()

            with autocast(device_type=str(device)):
                # forward
                logits = model(x=input_ids, mask=attention_mask)

                # shift for next-token prediction: logits[t] predicts labels = input_ids[t+1]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()

                # replace PAD tokens with ignore_index
                shift_labels[shift_labels == pad_id] = cross_entropy_ignore_index
                # compute loss
                loss = loss_fn(
                    shift_logits.view(-1, tokenizer.get_vocab_size()),
                    shift_labels.view(-1),
                )

            # scale gradient then backward
            scaler.scale(loss).backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), pretrain_config["clip_grad_max_norm"]
            )
            # step optimizer
            scaler.step(optimizer)
            # step scheduler
            scheduler.step()
            # update scaler
            scaler.update()
            # add train loss
            train_loss += loss.item()
            # show train info in tqdm
            batch_iter.set_postfix({"loss": f"{loss.item():6.3f}"})

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"validation"):
                # input
                input_ids = batch["input_ids"].to(device)
                attention_mask = get_attn_mask(input_ids, pad_id)

                with autocast(device_type=str(device)):
                    # forward
                    logits = model(x=input_ids, mask=attention_mask)

                    # shift for next-token prediction: logits[t] predicts labels = input_ids[t+1]
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = input_ids[:, 1:].contiguous()

                    # replace PAD tokens with ignore_index
                    shift_labels[shift_labels == pad_id] = cross_entropy_ignore_index
                    # compute loss
                    loss = loss_fn(
                        shift_logits.view(-1, tokenizer.get_vocab_size()),
                        shift_labels.view(-1),
                    )

                val_loss += loss.item()

        # average losses
        train_loss /= len(train_dl)
        val_loss /= len(val_dl)

        # perplexity metric
        train_ppl = float(np.exp(train_loss))
        val_ppl = float(np.exp(val_loss))

        # history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_perplexity"].append(train_ppl)
        history["val_perplexity"].append(val_ppl)

        # logging
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch+1} - {time_formatter(epoch_time)} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"train_ppl={train_ppl:.2f} | val_ppl={val_ppl:.2f}"
        )

        # callbacks
        model_dict = {
            "decoder": model.state_dict(),
        }
        optimizer_dict = {
            "adamw": optimizer.state_dict(),
        }

        stop_training = callbacks.step(
            monitor_value=val_ppl,
            epoch=epoch,
            model_dict=model_dict,
            optimizer_dict=optimizer_dict,
        )

        if stop_training:
            break

        print("\n")

    end_time = time.time()
    print(f"elapsed time: {time_formatter(end_time - start_time)}")
    return history
