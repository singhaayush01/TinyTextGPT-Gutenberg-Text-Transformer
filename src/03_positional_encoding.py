# Step 3: Positional Encoding (Rotary Positional Encoding)
# ------------------------------------------
# This tells our model *where* each token is in the sequence.

import torch
import torch.nn as nn

# Helper function: rotate half of the vector
def rotate_half(x):
    # splits the tensor into two halves (even and odd)
    x1, x2 = x.chunk(2, dim=-1)
    # concatenates them after rotating (swap and negate one half)
    return torch.cat((-x2, x1), dim=-1)

# Helper function: apply rotation with cosine and sine
def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

# Rotary Positional Encoding Class
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        """
        dim: size of the embedding (e.g., 128, 512, 768)
        max_seq_len: how long your input sequence can be
        """
        super().__init__()
        N = 10000  # frequency base
        # create inverse frequencies for each pair of dimensions
        inv_freq = 1. / (N ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(max_seq_len).float()

        # build the sinusoid matrix for all positions
        sinusoid_inp = torch.outer(position, inv_freq)
        self.register_buffer("cos", sinusoid_inp.cos())
        self.register_buffer("sin", sinusoid_inp.sin())

    def forward(self, x, seq_len=None):
        """
        x: shape [batch_size, seq_len, num_heads, head_dim]
        seq_len: actual length of the input sequence
        """
        if seq_len is None:
            seq_len = x.size(1)
        cos = self.cos[:seq_len].view(1, seq_len, 1, -1)
        sin = self.sin[:seq_len].view(1, seq_len, 1, -1)
        return apply_rotary_pos_emb(x, cos, sin)


# -------------------------------
#  Testing this module
if __name__ == "__main__":
    seq = torch.randn(1, 10, 4, 128)  # [batch, seq_len, heads, head_dim]
    rope = RotaryPositionalEncoding(dim=128)
    new_seq = rope(seq)
    print("Rotary Positional Encoding applied! Shape:", new_seq.shape)
