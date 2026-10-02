import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .layers import RMSNorm, TransformerBlock


class TinyTextGPT(nn.Module):
    """Small decoder-only GPT-style language model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(
            config.d_model, config.vocab_size, bias=False
        )

        # Tie input and output embeddings.
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight, mean=0.0, std=0.02
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight, mean=0.0, std=0.02
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError(
                f"Input length {input_ids.size(1)} exceeds "
                f"max_seq_len={self.config.max_seq_len}"
            )

        x = self.dropout(self.token_embedding(input_ids))
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        logits = self.lm_head(self.norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int | None = 50,
        top_p: float | None = 0.95,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()

        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        for _ in range(max_new_tokens):
            model_input = input_ids[:, -self.config.max_seq_len :]
            logits, _ = self(model_input)
            next_logits = logits[:, -1, :] / temperature

            if top_k is not None and top_k > 0:
                k = min(top_k, next_logits.size(-1))
                threshold = torch.topk(
                    next_logits, k, dim=-1
                ).values[:, -1, None]
                next_logits = next_logits.masked_fill(
                    next_logits < threshold, float("-inf")
                )

            probs = F.softmax(next_logits, dim=-1)

            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(
                    probs, descending=True, dim=-1
                )
                cumulative = torch.cumsum(
                    sorted_probs, dim=-1
                )
                remove = cumulative > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                sorted_probs = sorted_probs.masked_fill(
                    remove, 0.0
                )
                sorted_probs = sorted_probs / sorted_probs.sum(
                    dim=-1, keepdim=True
                )
                sampled_sorted = torch.multinomial(
                    sorted_probs, num_samples=1
                )
                next_token = sorted_indices.gather(
                    -1, sampled_sorted
                )
            else:
                next_token = torch.multinomial(
                    probs, num_samples=1
                )

            input_ids = torch.cat(
                [input_ids, next_token], dim=1
            )

            if (
                eos_token_id is not None
                and torch.all(
                    next_token.squeeze(-1) == eos_token_id
                )
            ):
                break

        return input_ids
