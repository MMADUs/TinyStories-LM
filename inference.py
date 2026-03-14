# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import torch
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from model import build_model, get_attn_mask


class Generator:
    """
    The generator system class encapsulates the model and tokenizer for inference.
    Provides a generate method to create text based on the provided context.

    Args:
    - config: model configuration dictionary
    - version: version string to identify the model checkpoint to load
    - context: initial dialogue context to prime the model with (optional)
    """

    def __init__(self, config, version, load_from_epoch):
        self.device = config["device"]

        output_dir = Path(config["output_dir_path"])

        # load tokenizer from disk
        tokenizer_filename = config["tokenizer_filename"]
        tokenizer_path = output_dir / tokenizer_filename

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))

        # TODO: this is temporary, should be removed soon when tokenizer is retrained
        self.tokenizer.decoder = ByteLevelDecoder()

        filename = config["pretraining"]["model_ckpt_filename"].format(
            version
        )  # base filename

        ckpt_filename = f"{filename}_epoch_{load_from_epoch}"  # epoch filename
        ckpt_filename = (
            ckpt_filename + config["pretraining"]["ckpt_format"]
        )  # full filename with format
        ckpt_path = output_dir / ckpt_filename

        print("loading model from:", ckpt_path)

        checkpoint = torch.load(ckpt_path)
        model_state = checkpoint["model"]

        self.model = build_model(config, self.tokenizer)
        self.model.load_state_dict(model_state)
        self.model = self.model.to(self.device)
        self.model.eval()

        # load special token ids
        special_tokens_dict = config["special_tokens"]

        self.bos_id = self.tokenizer.token_to_id(special_tokens_dict["begin"])
        self.eos_id = self.tokenizer.token_to_id(special_tokens_dict["end"])
        self.sep_id = self.tokenizer.token_to_id(special_tokens_dict["separator"])
        self.pad_id = self.tokenizer.token_to_id(special_tokens_dict["padding"])

    def generate(
        self,
        context: str = None,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> str:
        ctx_ids = self.tokenizer.encode(context).ids if context is not None else []

        # initalize input_ids with BOS + init_context
        input_ids = [self.bos_id] + ctx_ids

        input_ids = torch.tensor(
            input_ids, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                attention_mask = get_attn_mask(input_ids, self.pad_id).to(self.device)
                logits = self.model(x=input_ids, mask=attention_mask)
                next_token_logits = logits[:, -1, :]  # (1, vocab_size)

                # apply temperature
                next_token_logits = next_token_logits / temperature

                # top-k filtering
                if top_k > 0:
                    values, indices = torch.topk(next_token_logits, top_k, dim=-1)
                    filtered = torch.full_like(next_token_logits, float("-inf"))
                    filtered.scatter_(-1, indices, values)
                    next_token_logits = filtered

                # top-p (nucleus) filtering
                if top_p > 0.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    # remove tokens with cumulative prob above threshold
                    sorted_indices_to_remove = (
                        cumulative_probs - torch.softmax(sorted_logits, dim=-1) > top_p
                    )
                    sorted_logits[sorted_indices_to_remove] = float("-inf")
                    next_token_logits = sorted_logits.scatter(
                        -1, sorted_indices, sorted_logits
                    )

                # sample
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, 1)

                # append
                input_ids = torch.cat([input_ids, next_token_id], dim=1)

                # stop if EOS is generated
                if next_token_id.item() == self.eos_id:
                    break

        # decode
        list_ids = input_ids.squeeze(0).tolist()

        cleaned_ids = []

        for token_id in list_ids:
            if token_id in (self.bos_id, self.pad_id, self.sep_id):
                continue
            if token_id == self.eos_id:
                break
            cleaned_ids.append(token_id)

        generated_text = self.tokenizer.decode(cleaned_ids)

        cleaned_text = (
            generated_text.replace(" ,", ",")
            .replace(" .", ".")
            .replace(" !", "!")
            .replace(" ?", "?")
            .replace(" '", "'")
            .replace(" \n", "\n")
            .replace("Ġ", " ")
            .replace("Ċ", "\n")
            .strip()
        )

        return cleaned_text
