# Step 4: Grouped Query Attention (GQA)

import torch

from tinytextgpt.layers import GroupedQueryAttention


if __name__ == "__main__":
    x = torch.randn(2, 16, 128)
    gqa = GroupedQueryAttention(
        d_model=128,
        n_heads=8,
        n_kv_heads=2,
        max_seq_len=64,
    )
    y = gqa(x)
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
