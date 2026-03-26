# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

from pathlib import Path

import torch

from src.modules.decoder import build_model


def get_checkpoint_path(config, stage_config, version, load_from_epoch):
    """
    Get the checkpoint path for the given version and epoch number.
    """
    output_dir = Path(config["output_dir_path"])

    filename = stage_config["model_ckpt_filename"].format(version)  # base filename

    ckpt_filename = f"{filename}_epoch_{load_from_epoch}"  # epoch filename
    ckpt_filename = (
        ckpt_filename + stage_config["ckpt_format"]
    )  # full filename with format

    ckpt_path = output_dir / ckpt_filename

    return ckpt_path


def get_or_build_state(
    config, stage, tokenizer, version, initial_train, load_from_epoch
):
    """
    Get the model and optimizer state for training. If initial_train is True, build a new model and optimizer.
    Otherwise, load the model and optimizer state from the last checkpoint.
    """
    stage_config = config[stage]
    device = config["device"]

    additional_epochs = 0

    # if resume train but its fine tuning
    # load from pretrained -> fine tune pretrained
    if not initial_train and stage == "finetuning":
        pretrained_config = config["pretraining"]

        ckpt_path = get_checkpoint_path(
            config, pretrained_config, version, load_from_epoch
        )

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found at {ckpt_path}")

        print(
            f"Checkpoint found at {ckpt_path}, loading pretrained weights for finetuning"
        )

        # load only model weights, not optimizer state
        model, _unused_optimizer, _last_epoch = get_last_checkpoint_state(
            config,
            pretrained_config,
            tokenizer,
            version,
            load_from_epoch,
        )

        # fresh optimizer with finetuning lr
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=stage_config["lr"],
            weight_decay=stage_config["weight_decay"],
            fused=True,
        )

        return model, optimizer, 0  # epoch resets for finetuning

    # designed for pretrainining
    # also works for fine tuning but early fine tuning checkpoint must exist
    ckpt_path = get_checkpoint_path(config, stage_config, version, load_from_epoch)

    # initialize model and optimizer
    if initial_train or not ckpt_path.exists() or load_from_epoch is None:
        print("No checkpoint found, start initial training")

        model = build_model(config, tokenizer)
        model = model.to(device)
        # model = torch.compile(model)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=stage_config["lr"],
            weight_decay=stage_config["weight_decay"],
            fused=True,
        )
    else:
        print(f"Checkpoint found at {ckpt_path}, loading checkpoint for training")

        model, optimizer, last_epoch = get_last_checkpoint_state(
            config,
            stage_config,
            tokenizer,
            version,
            load_from_epoch,
        )
        additional_epochs = last_epoch + 1

    return model, optimizer, additional_epochs


def save_checkpoint_state(
    model, optimizer, scheduler, epoch, val_loss, val_ppl, config, stage_config, version
):
    """
    Save the checkpoint state for model and optimizer. This is used to resume training from the last checkpoint.
    """
    ckpt_state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "val_perplexity": val_ppl,
    }

    output_dir = Path(config["output_dir_path"])
    filename = stage_config["model_ckpt_filename"].format(version)  # base filename

    curr_ckpt = f"{filename}_epoch_{epoch+1}"  # epoch filename
    curr_ckpt = curr_ckpt + stage_config["ckpt_format"]  # full filename with format

    curr_ckpt_path = output_dir / curr_ckpt

    torch.save(ckpt_state_dict, curr_ckpt_path)

    print(f"Checkpoint saved at: {curr_ckpt_path}")


def get_last_checkpoint_state(
    config, stage_config, tokenizer, version, load_from_epoch
):
    """
    Load the last checkpoint state for model and optimizer. This is used to resume training from the last checkpoint.
    """
    device = config["device"]

    ckpt_path = get_checkpoint_path(config, stage_config, version, load_from_epoch)

    checkpoint = torch.load(ckpt_path)
    model_state = checkpoint["model"]
    optimizer_state = checkpoint["optimizer"]
    last_epoch = checkpoint["epoch"]

    val_loss = checkpoint["val_loss"]
    val_ppl = checkpoint["val_perplexity"]

    print(
        f"Checkpoint loaded from epoch {last_epoch+1} with val loss {val_loss:.4f} and val perplexity {val_ppl:.4f}"
    )

    model = build_model(config, tokenizer)
    model.load_state_dict(model_state)
    model = model.to(device)
    # model = torch.compile(model)

    lr = stage_config["lr"]
    weight_decay = stage_config["weight_decay"]

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, fused=True
    )
    optimizer.load_state_dict(optimizer_state)

    return model, optimizer, last_epoch
