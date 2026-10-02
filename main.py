import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tinytextgpt import ModelConfig, TinyTextGPT

def main():
    config = ModelConfig()
    model = TinyTextGPT(config)
    parameters = sum(p.numel() for p in model.parameters())

    print("TinyTextGPT is ready.")
    print(f"Parameters: {parameters:,}")
    print("Pipeline:")
    print("  python src/01_get_data.py")
    print("  python src/02_train_tokenizer.py")
    print("  python src/09_train_model.py")
    print('  python src/10_generate_text.py --prompt "Once upon a time"')

    x = torch.randint(0, config.vocab_size, (1, 8))
    logits, _ = model(x)
    print("Smoke-test logits:", tuple(logits.shape))

if __name__ == "__main__":
    main()
