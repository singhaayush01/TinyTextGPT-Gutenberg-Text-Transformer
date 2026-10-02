from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 10_000
    max_seq_len: int = 256
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 2
    num_experts: int = 4
    experts_per_token: int = 2
    ff_hidden_dim: int = 768
    dropout: float = 0.1
    rope_base: float = 10_000.0

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("attention head dimension must be even for RoPE")
        if not 1 <= self.experts_per_token <= self.num_experts:
            raise ValueError("experts_per_token must be between 1 and num_experts")
