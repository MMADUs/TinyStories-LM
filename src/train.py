# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import time
from typing import Optional, Literal

import numpy as np
import torch
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup

from src.modules.utils import get_attn_mask
from src.utils import time_formatter, set_random_seed, calculate_scheduler_steps
from src.serialization import get_or_build_state, save_checkpoint_state


def train_model(
    config,
    train_dl,
    val_dl,
    tokenizer,
    stage: Literal["pretraining", "finetuning"],
    version: str = "NA",
    initial_train: bool = True,
    load_from_epoch: Optional[int] = None,
) -> dict:
    """the training loop

    Args:
        config: model configuration dictionary
        train_dl: training dataloader
        val_dl: validation dataloader
        tokenizer: trained tokenizer
        stage: training stage ("pretraining" or "finetuning")
        version: version string to identify the checkpoint file (default: "NA")
        initial_train: whether to start training from scratch (default: True)
        load_from_epoch: epoch number to resume training from (default: None)

    Returns:
        history: the training history in dictionary
    """
    # set random seed for reproducibility
    set_random_seed(config["random_seed"])

    stage_config = config[stage]
    device = config["device"]
    pad_id = tokenizer.token_to_id(config["special_tokens"]["padding"])

    init_epoch = 0

    print("preparing training...")

    # initialize model and optimizer
    model, optimizer, additional_epochs = get_or_build_state(
        config, stage, tokenizer, version, initial_train, load_from_epoch
    )

    init_epoch = additional_epochs

    # learning rate scheduler: linear warmup + cosine annealing
    num_training_steps, num_warmup_steps = calculate_scheduler_steps(
        init_epoch, stage_config, len(train_dl)
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # criterion
    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=stage_config["cross_entropy_ignore_index"],
        label_smoothing=stage_config["label_smoothing"],
    )

    # # gradient scaler
    # scaler = GradScaler()

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_perplexity": [],
        "val_perplexity": [],
    }

    print("starting training...")

    start_time = time.time()

    for epoch in range(init_epoch, stage_config["num_epochs"]):
        epoch_start = time.time()
        torch.cuda.empty_cache()

        model.train()

        global_train_loss = 0.0
        train_loss = 0.0
        append_train_history_step = stage_config["append_train_history_step"]
        train_step = 0

        batch_iter = tqdm(train_dl, desc=f"epoch {epoch+1}")

        for batch in batch_iter:
            # input
            input_ids = batch["input_ids"].to(device)
            attention_mask = get_attn_mask(input_ids, pad_id)

            # reset optimizer gradient
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=str(device), dtype=torch.bfloat16):
                # forward
                logits, aux_loss = model(x=input_ids, mask=attention_mask)
                
                # TODO: handle aux loss

                # drop last token to shift logits
                shift_logits = logits[:, :-1, :].contiguous()

                # shift for next-token prediction: logits[t] predicts labels = input_ids[t+1]
                if stage == "pretraining":
                    # drop first token to shift labels
                    shift_labels = input_ids[:, 1:].contiguous()

                    # replace PAD tokens with ignore_index
                    shift_labels[shift_labels == pad_id] = stage_config[
                        "cross_entropy_ignore_index"
                    ]
                elif stage == "finetuning":
                    # labels PAD is already replaced with ignore_index in the dataset __getitem__
                    labels = batch["labels"].to(device)

                    # drop first token to shift labels
                    shift_labels = labels[:, 1:].contiguous()
                else:
                    raise ValueError(f"Invalid stage: {stage}")

                # compute loss
                loss = loss_fn(
                    shift_logits.view(-1, tokenizer.get_vocab_size()),
                    shift_labels.view(-1),
                )

            # scale gradient then backward
            # scaler.scale(loss).backward()
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), stage_config["clip_grad_max_norm"]
            )
            # step optimizer
            # scaler.step(optimizer)
            optimizer.step()
            # step scheduler
            scheduler.step()
            # # update scaler
            # scaler.update()

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
                train_step = 0

        model.eval()

        global_val_loss = 0.0
        val_loss = 0.0
        append_val_history_step = stage_config["append_val_history_step"]
        val_step = 0

        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"validation"):
                # input
                input_ids = batch["input_ids"].to(device)
                attention_mask = get_attn_mask(input_ids, pad_id)

                with autocast(device_type=str(device), dtype=torch.bfloat16):
                    # forward
                    logits, aux_loss = model(x=input_ids, mask=attention_mask)

                    # TODO: handle aux loss

                    # drop last token to shift logits
                    shift_logits = logits[:, :-1, :].contiguous()

                    # shift for next-token prediction: logits[t] predicts labels = input_ids[t+1]
                    if stage == "pretraining":
                        # drop first token to shift labels
                        shift_labels = input_ids[:, 1:].contiguous()

                        # replace PAD tokens with ignore_index
                        shift_labels[shift_labels == pad_id] = stage_config[
                            "cross_entropy_ignore_index"
                        ]
                    elif stage == "finetuning":
                        # labels PAD is already replaced with ignore_index in the dataset __getitem__
                        labels = batch["labels"].to(device)

                        # drop first token to shift labels
                        shift_labels = labels[:, 1:].contiguous()
                    else:
                        raise ValueError(f"Invalid stage: {stage}")

                    # compute loss
                    loss = loss_fn(
                        shift_logits.view(-1, tokenizer.get_vocab_size()),
                        shift_labels.view(-1),
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
                    val_step = 0

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
        save_checkpoint_state(
            model,
            optimizer,
            scheduler,
            epoch,
            global_val_loss,
            val_ppl,
            config,
            stage_config,
            version,
        )

        print("\n")

    end_time = time.time()
    print(f"elapsed time: {time_formatter(end_time - start_time)}")
    return history
