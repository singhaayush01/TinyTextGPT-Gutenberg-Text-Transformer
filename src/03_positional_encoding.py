# Step 3: Rotary Positional Encoding (RoPE)
# -----------------------------------------
# RoPE rotates query/key features according to token position.

import torch
import torch.nn as nn


def rotate_half(x):
    """Rotate adjacent pairs: (x0, x1) -> (-x1, x0)."""
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack(
        (-x_odd, x_even), dim=-1
    ).flatten(-2)


def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)


class RotaryPositionalEncoding(nn.Module):
    def __init__(
        self,
        dim,
        max_seq_len=512,
        base=10_000.0,
    ):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError(
                "RoPE dimension must be even"
            )

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(
                    0, dim, 2
                ).float()
                / dim
            )
        )
        position = torch.arange(
            max_seq_len,
            dtype=torch.float32,
        )
        freqs = torch.outer(
            position, inv_freq
        )

        # Duplicate each angle so cos/sin match head_dim.
        angles = torch.repeat_interleave(
            freqs, 2, dim=-1
        )
        self.register_buffer(
            "cos", angles.cos()
        )
        self.register_buffer(
            "sin", angles.sin()
        )

    def forward(self, x, seq_len=None):
        """
        x: [batch, seq_len, num_heads, head_dim]
        """
        if seq_len is None:
            seq_len = x.size(1)

        cos = self.cos[:seq_len].to(
            dtype=x.dtype
        ).view(1, seq_len, 1, -1)

        sin = self.sin[:seq_len].to(
            dtype=x.dtype
        ).view(1, seq_len, 1, -1)

        return apply_rotary_pos_emb(
            x, cos, sin
        )


if __name__ == "__main__":
    seq = torch.randn(
        1, 10, 4, 128
    )
    rope = RotaryPositionalEncoding(
        dim=128
    )
    new_seq = rope(seq)
    print(
        "RoPE applied! Shape:",
        new_seq.shape,
    )
