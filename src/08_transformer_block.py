# Step 8: Full Transformer Block

import torch

from tinytextgpt import ModelConfig
from tinytextgpt.layers import TransformerBlock


if __name__ == "__main__":
    config = ModelConfig(
        vocab_size=10_000,
        max_seq_len=64,
        d_model=128,
        n_layers=2,
        n_heads=8,
        n_kv_heads=2,
        num_experts=4,
        experts_per_token=2,
        ff_hidden_dim=384,
        dropout=0.0,
    )

    block = TransformerBlock(config)
    x = torch.randn(2, 16, config.d_model)
    y = block(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
