def get_last_checkpoint_state(config, tokenizer, version):
    """
    Load the last checkpoint state for model and optimizer. This is used to resume training from the last checkpoint.

    Args:
    - config: model configuration dictionary
    - tokenizer: trained tokenizer to build the model
    - version: version string to identify the checkpoint file
    """
    device = config["device"]

    output_dir = Path(config["output_dir_path"])
    ckpt_filename = config["model_ckpt_filename"].format(version)
    ckpt_path = output_dir / ckpt_filename

    checkpoint = torch.load(ckpt_path)
    model_state = checkpoint["model"]["decoder"]
    optimizer_state = checkpoint["optimizer"]["adamw"]
    last_epoch = checkpoint["epoch"]

    model = build_model(config, tokenizer)
    model.load_state_dict(model_state)
    model = model.to(device)

    lr = config["lr"]
    weight_decay = config["weight_decay"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer.load_state_dict(optimizer_state)

    return model, optimizer, last_epoch