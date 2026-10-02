import argparse
import math
import os
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader, random_split
from tokenizers import Tokenizer

from .config import ModelConfig
from .data import GutenbergTokenDataset
from .model import TinyTextGPT


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def evaluate(model, loader, device, max_batches=20):
    model.eval()
    total = 0.0
    count = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, targets=y)
            total += loss.item()
            count += 1

            if count >= max_batches:
                break

    return total / max(count, 1)


def train_model(args):
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    tokenizer = Tokenizer.from_file(args.tokenizer)

    config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        max_seq_len=args.block_size,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        n_kv_heads=args.kv_heads,
        num_experts=args.experts,
        experts_per_token=args.experts_per_token,
        ff_hidden_dim=args.ff_hidden,
        dropout=args.dropout,
    )

    dataset = GutenbergTokenDataset(
        args.data_dir,
        args.tokenizer,
        args.block_size,
    )

    val_size = max(
        1, int(len(dataset) * args.val_fraction)
    )
    train_size = len(dataset) - val_size

    if train_size < 1:
        raise ValueError(
            "Not enough training chunks. Reduce block_size "
            "or val_fraction."
        )

    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = TinyTextGPT(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(
        1, int(total_steps * 0.05)
    )

    def lr_multiplier(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps

        progress = (
            step - warmup_steps
        ) / max(1, total_steps - warmup_steps)

        return 0.1 + 0.9 * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_multiplier
    )

    os.makedirs(
        os.path.dirname(args.output) or ".",
        exist_ok=True,
    )

    print(f"Device: {device}")
    print(
        f"Training chunks: {train_size:,} | "
        f"Validation chunks: {val_size:,}"
    )
    print(
        "Parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        for batch_idx, (x, y) in enumerate(
            train_loader, start=1
        ):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, targets=y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip
            )
            optimizer.step()
            scheduler.step()

            running += loss.item()

            if (
                batch_idx % args.log_every == 0
                or batch_idx == len(train_loader)
            ):
                print(
                    f"epoch {epoch}/{args.epochs} | "
                    f"batch {batch_idx}/{len(train_loader)} | "
                    f"train loss {running / batch_idx:.4f} | "
                    f"lr {scheduler.get_last_lr()[0]:.2e}"
                )

        val_loss = evaluate(
            model, val_loader, device
        )
        print(
            f"epoch {epoch}: validation loss "
            f"{val_loss:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "config": asdict(config),
                    "val_loss": val_loss,
                    "epoch": epoch,
                },
                args.output,
            )
            print(
                "Saved best checkpoint -> "
                f"{args.output}"
            )

    return model


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Train TinyTextGPT on Project Gutenberg text"
        )
    )
    parser.add_argument(
        "--data-dir", default="data/raw"
    )
    parser.add_argument(
        "--tokenizer",
        default="gutenberg_tokenizer.json",
    )
    parser.add_argument(
        "--output",
        default="checkpoints/tinytextgpt.pt",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, mps, ...",
    )
    parser.add_argument(
        "--epochs", type=int, default=3
    )
    parser.add_argument(
        "--batch-size", type=int, default=8
    )
    parser.add_argument(
        "--block-size", type=int, default=128
    )
    parser.add_argument(
        "--d-model", type=int, default=256
    )
    parser.add_argument(
        "--layers", type=int, default=4
    )
    parser.add_argument(
        "--heads", type=int, default=8
    )
    parser.add_argument(
        "--kv-heads", type=int, default=2
    )
    parser.add_argument(
        "--experts", type=int, default=4
    )
    parser.add_argument(
        "--experts-per-token",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--ff-hidden", type=int, default=768
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--grad-clip", type=float, default=1.0
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    parser.add_argument(
        "--log-every", type=int, default=50
    )
    return parser


def main():
    train_model(build_parser().parse_args())


if __name__ == "__main__":
    main()
