# Step 6: Mixture-of-Experts Feed-Forward Layer

import torch

from tinytextgpt.layers import MixtureOfExperts


if __name__ == "__main__":
    x = torch.randn(2, 16, 128)
    moe = MixtureOfExperts(
        d_model=128,
        hidden_dim=384,
        num_experts=4,
        top_k=2,
    )
    y = moe(x)
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
