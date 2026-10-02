from pathlib import Path

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class GutenbergTokenDataset(Dataset):
    """Turn Project Gutenberg text into fixed next-token training chunks."""

    def __init__(
        self,
        data_dir: str,
        tokenizer_path: str,
        block_size: int = 256,
    ):
        self.block_size = block_size
        tokenizer = Tokenizer.from_file(tokenizer_path)

        eos_id = tokenizer.token_to_id("[eos]")
        if eos_id is None:
            raise ValueError(
                "Tokenizer is missing required [eos] token"
            )

        files = sorted(Path(data_dir).glob("*.txt"))
        if not files:
            raise FileNotFoundError(
                f"No .txt files found in {data_dir}. "
                "Run: python src/01_get_data.py"
            )

        all_ids: list[int] = []
        for path in files:
            text = path.read_text(
                encoding="utf-8", errors="replace"
            )
            all_ids.extend(tokenizer.encode(text).ids)
            all_ids.append(eos_id)

        self.tokens = torch.tensor(
            all_ids, dtype=torch.long
        )
        self.num_samples = max(
            0, (len(self.tokens) - 1) // block_size
        )

        if self.num_samples == 0:
            raise ValueError(
                "Dataset is too small for this block_size"
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        start = index * self.block_size
        chunk = self.tokens[
            start : start + self.block_size + 1
        ]
        return chunk[:-1], chunk[1:]
