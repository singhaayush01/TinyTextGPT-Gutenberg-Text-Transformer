# Step 7: RMSNorm and Residual Connections

import torch

from tinytextgpt.layers import RMSNorm


if __name__ == "__main__":
    x = torch.randn(2, 16, 128)
    norm = RMSNorm(128)
    normalized = norm(x)

    # Residual connections add a transformed tensor
    # back to the original representation.
    residual_example = x + normalized

    print("Normalized shape:", normalized.shape)
    print("Residual shape  :", residual_example.shape)
