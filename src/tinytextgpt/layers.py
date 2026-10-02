from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent pairs: (x0, x1) -> (-x1, x0)."""
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE) cache."""

    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10_000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE dimension must be even")

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        angles = torch.repeat_interleave(freqs, 2, dim=-1)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, heads, head_dim]
        seq_len = x.size(1)
        if seq_len > self.cos.size(0):
            raise ValueError(
                f"Sequence length {seq_len} exceeds RoPE cache {self.cos.size(0)}"
            )
        cos = self.cos[:seq_len].to(dtype=x.dtype).view(1, seq_len, 1, -1)
        sin = self.sin[:seq_len].to(dtype=x.dtype).view(1, seq_len, 1, -1)
        return (x * cos) + (rotate_half(x) * sin)


def causal_mask(seq_len: int, device=None) -> torch.Tensor:
    """Boolean lower-triangular mask. True means attention is allowed."""
    return torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    )


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention with RoPE and causal masking."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
        rope_base: float = 10_000.0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.kv_repeat = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryEmbedding(
            self.head_dim, max_seq_len=max_seq_len, base=rope_base
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(
            batch, seq_len, self.n_heads, self.head_dim
        )
        k = self.k_proj(x).view(
            batch, seq_len, self.n_kv_heads, self.head_dim
        )
        v = self.v_proj(x).view(
            batch, seq_len, self.n_kv_heads, self.head_dim
        )

        q = self.rope(q)
        k = self.rope(k)

        # Each key/value head is shared by a group of query heads.
        k = k.repeat_interleave(self.kv_repeat, dim=2)
        v = v.repeat_interleave(self.kv_repeat, dim=2)

        q = q.transpose(1, 2)  # [B, H, T, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        allowed = causal_mask(seq_len, x.device).view(
            1, 1, seq_len, seq_len
        )

        if attention_mask is not None:
            key_mask = attention_mask.to(torch.bool).view(
                batch, 1, 1, seq_len
            )
            allowed = allowed & key_mask

        scores = scores.masked_fill(
            ~allowed, torch.finfo(scores.dtype).min
        )
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(
            batch, seq_len, -1
        )
        return self.out_proj(out)


class SwiGLUExpert(nn.Module):
    def __init__(
        self, d_model: int, hidden_dim: int, dropout: float = 0.0
    ):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.up = nn.Linear(d_model, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.gate(x)) * self.up(x)
        return self.down(self.dropout(hidden))


class MixtureOfExperts(nn.Module):
    """
    Educational top-k Mixture-of-Experts layer.

    The router selects the top experts per token and combines their outputs.
    For readability every expert is evaluated; production MoE systems normally
    use sparse token dispatch for greater efficiency.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts")

        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                SwiGLUExpert(d_model, hidden_dim, dropout)
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        router_logits = self.router(x)
        top_values, top_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )
        top_weights = F.softmax(top_values, dim=-1)

        output = torch.zeros_like(x)
        for expert_id, expert in enumerate(self.experts):
            expert_weight = (
                (top_indices == expert_id).to(x.dtype) * top_weights
            ).sum(dim=-1, keepdim=True)
            output = output + expert(x) * expert_weight
        return output


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * x * torch.rsqrt(rms + self.eps)


class TransformerBlock(nn.Module):
    """Pre-norm decoder block with GQA, MoE and residual connections."""

    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = GroupedQueryAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            rope_base=config.rope_base,
        )
        self.moe_norm = RMSNorm(config.d_model)
        self.moe = MixtureOfExperts(
            d_model=config.d_model,
            hidden_dim=config.ff_hidden_dim,
            num_experts=config.num_experts,
            top_k=config.experts_per_token,
            dropout=config.dropout,
        )
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.resid_dropout(
            self.attn(
                self.attn_norm(x),
                attention_mask=attention_mask,
            )
        )
        x = x + self.resid_dropout(self.moe(self.moe_norm(x)))
        return x
