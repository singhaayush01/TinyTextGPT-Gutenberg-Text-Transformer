# Step 5: Causal Masking

from tinytextgpt.layers import causal_mask


if __name__ == "__main__":
    print(
        causal_mask(8).int()
    )
