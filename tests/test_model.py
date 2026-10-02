import sys
from pathlib import Path

import torch

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from tinytextgpt import ModelConfig, TinyTextGPT
from tinytextgpt.layers import (
    GroupedQueryAttention,
    MixtureOfExperts,
    RMSNorm,
    RotaryEmbedding,
    causal_mask,
)


def tiny_config():
    return ModelConfig(
        vocab_size=101,
        max_seq_len=32,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        num_experts=4,
        experts_per_token=2,
        ff_hidden_dim=128,
        dropout=0.0,
    )


def test_rope_preserves_shape():
    rope = RotaryEmbedding(
        dim=16, max_seq_len=32
    )
    x = torch.randn(2, 10, 4, 16)
    assert rope(x).shape == x.shape


def test_causal_mask_is_lower_triangular():
    mask = causal_mask(4)
    expected = torch.tensor(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(mask, expected)


def test_gqa_preserves_shape():
    attn = GroupedQueryAttention(
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        dropout=0.0,
    )
    x = torch.randn(2, 12, 64)
    assert attn(x).shape == x.shape


def test_gqa_does_not_look_into_future():
    torch.manual_seed(0)
    attn = GroupedQueryAttention(
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        dropout=0.0,
    )
    attn.eval()

    x1 = torch.randn(1, 10, 64)
    x2 = x1.clone()
    x2[:, 6:] = torch.randn_like(x2[:, 6:]) * 20

    y1 = attn(x1)
    y2 = attn(x2)

    assert torch.allclose(
        y1[:, :6],
        y2[:, :6],
        atol=1e-5,
        rtol=1e-5,
    )


def test_moe_and_rmsnorm_preserve_shape():
    x = torch.randn(2, 10, 64)
    moe = MixtureOfExperts(
        d_model=64,
        hidden_dim=128,
        num_experts=4,
        top_k=2,
    )
    norm = RMSNorm(64)

    assert moe(x).shape == x.shape
    assert norm(x).shape == x.shape


def test_model_forward_and_backward():
    model = TinyTextGPT(tiny_config())
    x = torch.randint(0, 101, (2, 16))
    y = torch.randint(0, 101, (2, 16))

    logits, loss = model(x, targets=y)

    assert logits.shape == (2, 16, 101)
    assert loss is not None
    loss.backward()

    assert any(
        p.grad is not None
        for p in model.parameters()
        if p.requires_grad
    )


def test_generation_adds_tokens():
    model = TinyTextGPT(tiny_config())
    prompt = torch.randint(0, 101, (1, 5))

    result = model.generate(
        prompt,
        max_new_tokens=4,
        temperature=1.0,
        top_k=20,
        top_p=0.9,
    )

    assert result.shape == (1, 9)
