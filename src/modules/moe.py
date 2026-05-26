# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFeedForward(nn.Module):
    """feed forward network with SwiGLU activation.

    Args:
        d_model: dimension of the model (embedding size)
        d_ff: dimension of the feed forward network
        dropout: dropout rate
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()

        self.proj1 = nn.Linear(d_model, d_ff * 2)  # gating
        self.proj2 = nn.Linear(d_ff, d_model)  # output projection

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, d_model)

        # x1, x2: (batch, seq_len, d_ff)
        x1, x2 = self.proj1(x).chunk(2, dim=-1)

        # SwiGLU activation
        x = F.silu(x1) * x2

        # out: (batch, seq_len, d_model)
        out = self.proj2(x)
        return self.dropout(out)


class MoERouter(nn.Module):
    """
    Routes input tokens to top-k experts.

    Args:
    - d_model: dimension of the model
    - num_experts: total number of experts
    - top_k: number of experts to route each token to
    """

    def __init__(self, d_model: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        gate_logits = self.gate(x)  # (batch, seq_len, num_experts)

        # top-k experts per token
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        # (batch, seq_len, top_k)

        # softmax only over selected top-k (sparse)
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        # (batch, seq_len, top_k)

        return top_k_weights, top_k_indices


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts layer using SwiGLU feed-forward experts.
    Replaces a standard FFN layer in a transformer block.

    Args:
    - d_model: dimension of the model
    - d_ff: dimension of each expert's feed-forward network
    - dropout: dropout rate
    - num_experts: total number of experts
    - top_k: number of experts each token is routed to
    - aux_loss_coef: coefficient for auxiliary load balancing loss
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float,
        num_experts: int = 8,
        top_k: int = 2,
        aux_loss_coef: float = 0.01,
    ):
        super().__init__()
        assert top_k <= num_experts, "top_k cannot exceed num_experts"

        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef

        self.router = MoERouter(d_model, num_experts, top_k)
        self.experts = nn.ModuleList(
            [SwiGLUFeedForward(d_model, d_ff, dropout) for _ in range(num_experts)]
        )

    def _aux_loss(self, top_k_indices, batch, seq_len):
        """
        Load balancing loss: penalizes uneven expert utilization.
        From Switch Transformer: encourages uniform distribution across experts.
        """
        # fraction of tokens routed to each expert
        flat_indices = top_k_indices.view(-1)  # (batch * seq_len * top_k,)
        total_tokens = batch * seq_len * self.top_k

        expert_counts = torch.bincount(flat_indices, minlength=self.num_experts).float()
        fraction_routed = expert_counts / total_tokens  # (num_experts,)

        # uniform target: each expert should get 1/num_experts of tokens
        uniform = torch.full_like(fraction_routed, 1.0 / self.num_experts)

        aux_loss = self.num_experts * (fraction_routed * uniform).sum()
        return aux_loss

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch, seq_len, d_model = x.shape

        top_k_weights, top_k_indices = self.router(x)
        # top_k_weights:  (batch, seq_len, top_k)
        # top_k_indices:  (batch, seq_len, top_k)

        # output accumulator
        out = torch.zeros_like(x)

        for k in range(self.top_k):
            expert_indices = top_k_indices[..., k]  # (batch, seq_len)
            expert_weights = top_k_weights[..., k]  # (batch, seq_len)

            for e in range(self.num_experts):
                # mask: which token positions are routed to expert e
                mask = expert_indices == e  # (batch, seq_len)

                if not mask.any():
                    continue

                # extract tokens for this expert
                expert_input = x[mask]  # (num_selected, d_model)

                # run through expert (unsqueeze seq dim since FFN expects 3D)
                expert_output = self.experts[e](expert_input.unsqueeze(1)).squeeze(1)
                # (num_selected, d_model)

                # weighted sum back
                weight = expert_weights[mask].unsqueeze(-1)  # (num_selected, 1)
                out[mask] += expert_output * weight

        # auxiliary load balancing loss
        aux_loss = self._aux_loss(top_k_indices, batch, seq_len)

        return out, aux_loss * self.aux_loss_coef
