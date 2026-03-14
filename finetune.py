# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup

from model import build_model, get_attn_mask
from utils import time_formatter, set_random_seed, get_last_checkpoint_state


def finetune_model(
    config,
    train_dl,
    val_dl,
    tokenizer,
    version: str = "NA",
    initial_train: bool = True,
) -> dict:
    """
    The fine-tuning loop.

    Args:
    - config: model configuration dictionary
    - train_dl: training dataloader
    - val_dl: validation dataloader
    - tokenizer: trained tokenizer
    - version: version string to identify the checkpoint file (default: "NA")
    - initial_train: whether to start training from scratch (default: True)
    """
    # set random seed for reproducibility
    set_random_seed(config["random_seed"])

    finetune_config = config["finetuning"]
    device = config["device"]

    pad_id = tokenizer.token_to_id(config["special_tokens"]["padding"])

    output_dir = Path(config["output_dir_path"])
    ckpt_filename = (
        finetune_config["model_ckpt_filename"].format(version)
        + finetune_config["ckpt_format"]
    )
    ckpt_path = output_dir / ckpt_filename

    init_epoch = 0

    # initialize model and optimizer
    if initial_train or not ckpt_path.exists():
        print("No checkpoint found, start initial training")

        model = build_model(config, tokenizer)
        model = model.to(device)

        lr = finetune_config["lr"]
        weight_decay = finetune_config["weight_decay"]

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    else:
        print(f"Checkpoint found at {ckpt_path}, loading checkpoint for training")

        model, optimizer, last_epoch = get_last_checkpoint_state(
            config, finetune_config, tokenizer, version
        )
        init_epoch = last_epoch + 1

    print("preparing training...")

    # warmup + linear decay learning rate scheduler
    lr_warmup_percentage = finetune_config["lr_warmup_percentage"]
    remaining_epochs = finetune_config["num_epochs"] - init_epoch
    num_training_steps = len(train_dl) * remaining_epochs
    num_warmup_steps = int(lr_warmup_percentage * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # criterion
    label_smoothing = finetune_config["label_smoothing"]
    cross_entropy_ignore_index = finetune_config["cross_entropy_ignore_index"]

    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=cross_entropy_ignore_index,  # ignore index for padding
        label_smoothing=label_smoothing,  # label smoothing factor
    )

    # gradient scaler
    scaler = GradScaler()

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_perplexity": [],
        "val_perplexity": [],
    }

    print("starting training...")

    start_time = time.time()

    for epoch in range(init_epoch, finetune_config["num_epochs"]):
        epoch_start = time.time()
        torch.cuda.empty_cache()

        model.train()

        global_train_loss = 0.0
        train_loss = 0.0
        append_train_history_step = finetune_config["append_train_history_step"]
        train_step = 0

        batch_iter = tqdm(train_dl, desc=f"epoch {epoch+1}")

        for batch in batch_iter:
            # input
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            attention_mask = get_attn_mask(input_ids, pad_id)

            # reset optimizer gradient
            optimizer.zero_grad()

            with autocast(device_type=str(device)):
                # forward
                logits = model(x=input_ids, mask=attention_mask)

                # compute loss directly with labels
                loss = loss_fn(
                    logits.view(-1, tokenizer.get_vocab_size()), labels.view(-1)
                )

            # scale gradient then backward
            scaler.scale(loss).backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), finetune_config["clip_grad_max_norm"]
            )
            # step optimizer
            scaler.step(optimizer)
            # step scheduler
            scheduler.step()
            # update scaler
            scaler.update()

            # add train loss
            train_loss += loss.item()
            global_train_loss += loss.item()

            # show real-time train info in tqdm
            batch_iter.set_postfix({"loss": f"{loss.item():6.3f}"})

            train_step += 1

            if train_step % append_train_history_step == 0:
                # avg loss and ppl for current step
                train_loss_avg = train_loss / train_step
                train_ppl = float(np.exp(train_loss_avg))
                # append to history
                history["train_loss"].append(train_loss_avg)
                history["train_perplexity"].append(train_ppl)
                # reset
                train_loss = 0.0

        model.eval()

        global_val_loss = 0.0
        val_loss = 0.0
        append_val_history_step = finetune_config["append_val_history_step"]
        val_step = 0

        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"validation"):
                # input
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                attention_mask = get_attn_mask(input_ids, pad_id)

                with autocast(device_type=str(device)):
                    # forward
                    logits = model(x=input_ids, mask=attention_mask)

                    # compute loss directly with labels
                    loss = loss_fn(
                        logits.view(-1, tokenizer.get_vocab_size()), labels.view(-1)
                    )

                # add val loss
                val_loss += loss.item()
                global_val_loss += loss.item()

                val_step += 1

                if val_step % append_val_history_step == 0:
                    # avg loss and ppl for current step
                    val_loss_avg = val_loss / val_step
                    val_ppl = float(np.exp(val_loss_avg))
                    # append to history
                    history["val_loss"].append(val_loss_avg)
                    history["val_perplexity"].append(val_ppl)
                    # reset
                    val_loss = 0.0

        # overall loss in avg
        global_train_loss /= len(train_dl)
        global_val_loss /= len(val_dl)

        # perplexity metric
        train_ppl = float(np.exp(global_train_loss))
        val_ppl = float(np.exp(global_val_loss))

        # logging
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch+1} - {time_formatter(epoch_time)} | "
            f"train_loss={global_train_loss:.6f} | val_loss={global_val_loss:.6f} | "
            f"train_ppl={train_ppl:.2f} | val_ppl={val_ppl:.2f}"
        )

        # checkpointing
        ckpt_state_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "val_loss": global_val_loss,
            "val_perplexity": val_ppl,
        }

        torch.save(ckpt_state_dict, ckpt_path)

        print("\n")

    end_time = time.time()
    print(f"elapsed time: {time_formatter(end_time - start_time)}")
    return history
