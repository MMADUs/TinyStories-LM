# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import torch
from pathlib import Path
from tokenizers import Tokenizer

from model import build_model, generate_causal_mask


class DialogueSystem:
    """
    The dialog system class encapsulates the model and tokenizer for inference. 
    Provides a chat method to generate responses given a prompt.

    Args:
    - config: model configuration dictionary
    - version: version string to identify the model checkpoint to load
    - emotion: optional emotion string to condition the response generation (default: "sentimental")
    """

    def __init__(self, config, version, emotion="sentimental"):
        self.version = version
        self.emotion = emotion

        self.device = config["device"]
        self.max_len = config["estimated_seq_len"]

        output_dir = Path(config["output_dir_path"])

        # load tokenizer from disk
        tokenizer_filename = config["tokenizer_filename"]
        tokenizer_path = output_dir / tokenizer_filename

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))

        # load model checkpoint from disk
        ckpt_filename = config["model_ckpt_filename"].format(version)
        ckpt_path = output_dir / ckpt_filename

        checkpoint = torch.load(ckpt_path)
        model_state = checkpoint["model"]["decoder"]

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

    def chat(self, prompt: str) -> str:
        # encode context and prompt
        ctx_ids = self.tokenizer.encode(self.emotion).ids if self.emotion else []
        p_ids = self.tokenizer.encode(prompt).ids

        # concatenate full sequence: [BOS] context [SEP] prompt [SEP]
        input_ids = [self.bos_id] + ctx_ids + [self.sep_id] + p_ids + [self.sep_id]

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        input_ids = input_ids.unsqueeze(0)  # add batch dimension

        with torch.no_grad():
            while input_ids.size(1) < self.max_len:
                # build attention mask
                padding_mask = (
                    (input_ids != self.pad_id).unsqueeze(1).unsqueeze(2)
                )  # shape -> (B,1,1,L)
                causal_mask = generate_causal_mask(
                    input_ids.size(1), self.device
                )  # shape -> (1,1,L,L)
                attention_mask = (
                    padding_mask & causal_mask
                )  # combine padding and causal masks

                # forward pass
                logits = self.model(x=input_ids, mask=attention_mask)
                next_token_logits = logits[:, -1, :]

                # # greedy decoding: pick the token with highest probability
                # next_token_id = torch.argmax(next_token_logits, dim=-1)

                # decode with top-k sampling
                top_k = 50
                values, indices = torch.topk(next_token_logits, top_k, dim=-1)  # (1, top_k)
                probs = torch.softmax(values, dim=-1)

                # sample: torch.multinomial expects 2D tensor (batch, classes)
                sampled = torch.multinomial(probs, 1)  # (1,1)

                # pick the actual token ids
                next_token_id = indices.gather(-1, sampled)  # (1,1)

                # append
                input_ids = torch.cat([input_ids, next_token_id], dim=1)

                # stop if EOS token reached
                if next_token_id.item() == self.eos_id:
                    break

        # decode generated token ids to text
        list_ids = input_ids.squeeze(0).tolist()  # remove batch dimension

        # find SEP positions
        sep_positions = [i for i, t in enumerate(list_ids) if t == self.sep_id]

        first_sep = sep_positions[0]
        second_sep = sep_positions[1]

        # extract token groups
        context_ids = list_ids[1:first_sep]                     # after BOS until SEP1
        prompt_ids  = list_ids[first_sep+1:second_sep]          # after SEP1 until SEP2
        response_ids = []

        # everything after SEP2 until EOS
        for tid in list_ids[second_sep+1:]:
            if tid == self.eos_id:
                break
            response_ids.append(tid)

        # decode
        ctx = self.tokenizer.decode(context_ids)
        pr  = self.tokenizer.decode(prompt_ids)
        rsp = self.tokenizer.decode(response_ids)

        # clean BPE artifacts
        ctx = ctx.replace("Ġ", "")
        pr  = pr.replace("Ġ", "")
        rsp = rsp.replace("Ġ", "")

        return ctx, pr, rsp
