import argparse

import torch
from tokenizers import Tokenizer

from .config import ModelConfig
from .model import TinyTextGPT
from .train import choose_device


def load_model(
    checkpoint_path: str,
    device: torch.device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )
    config = ModelConfig(**checkpoint["config"])
    model = TinyTextGPT(config).to(device)
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()
    return model


def generate_text(
    checkpoint: str,
    tokenizer_path: str,
    prompt: str,
    max_new_tokens: int = 120,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    device_name: str = "auto",
):
    device = choose_device(device_name)
    tokenizer = Tokenizer.from_file(
        tokenizer_path
    )
    model = load_model(checkpoint, device)

    prompt_ids = tokenizer.encode(prompt).ids
    if not prompt_ids:
        raise ValueError(
            "Prompt produced no tokens"
        )

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=tokenizer.token_to_id(
            "[eos]"
        ),
    )

    return tokenizer.decode(
        output_ids[0].tolist(),
        skip_special_tokens=True,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Generate text with a trained "
            "TinyTextGPT checkpoint"
        )
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/tinytextgpt.pt",
    )
    parser.add_argument(
        "--tokenizer",
        default="gutenberg_tokenizer.json",
    )
    parser.add_argument(
        "--prompt",
        default="Once upon a time",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--top-k", type=int, default=50
    )
    parser.add_argument(
        "--top-p", type=float, default=0.95
    )
    parser.add_argument(
        "--device", default="auto"
    )
    return parser


def main():
    args = build_parser().parse_args()
    print(
        generate_text(
            checkpoint=args.checkpoint,
            tokenizer_path=args.tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device_name=args.device,
        )
    )


if __name__ == "__main__":
    main()
