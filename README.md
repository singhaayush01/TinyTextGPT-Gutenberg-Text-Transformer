# TinyTextGPT — Gutenberg Text Transformer 🧠

A from-scratch decoder-only Transformer language model built with PyTorch and trained on public-domain text from Project Gutenberg.

The project is educational: each numbered script introduces one major concept used by modern GPT-style models, while the reusable implementation lives in `src/tinytextgpt/`.

## Architecture

TinyTextGPT includes:
- Byte-Level BPE tokenizer
- Rotary Positional Encoding (RoPE)
- Grouped Query Attention (GQA)
- Causal self-attention masking
- SwiGLU Mixture-of-Experts (MoE) feed-forward layer
- RMSNorm and residual connections
- Stacked decoder-only Transformer blocks
- Next-token training with AdamW
- Temperature, top-k, and top-p text generation

> The MoE layer is intentionally readability-first: it evaluates all experts, then applies top-k routing weights. Production MoE systems use sparse dispatch.

## 10-Step Roadmap

1. ✅ Get the Data — download Project Gutenberg books.
2. ✅ Train the Tokenizer — train a Byte-Level BPE tokenizer.
3. ✅ Positional Encoding — apply RoPE to queries and keys.
4. ✅ Grouped Query Attention — share key/value heads across query-head groups.
5. ✅ Causal Masking — prevent attention to future tokens.
6. ✅ Feed-Forward MoE — route each token to its top experts.
7. ✅ Normalization & Skip Connections — RMSNorm + residual paths.
8. ✅ Full Transformer Block — combine attention, MoE, norms, and residuals.
9. ✅ Training Loop — next-token prediction with validation and checkpoints.
10. ✅ Text Generation — autoregressive sampling with top-k and top-p.

## Project Structure

```text
.
├── main.py
├── requirements.txt
├── src/
│   ├── 01_get_data.py
│   ├── 02_train_tokenizer.py
│   ├── 03_positional_encoding.py
│   ├── 04_grouped_query_attention.py
│   ├── 05_causal_masking.py
│   ├── 06_moe_feed_forward.py
│   ├── 07_norm_and_residuals.py
│   ├── 08_transformer_block.py
│   ├── 09_train_model.py
│   ├── 10_generate_text.py
│   └── tinytextgpt/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── generate.py
│       ├── layers.py
│       ├── model.py
│       └── train.py
└── tests/
    └── test_model.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:
```powershell
.venv\Scripts\activate
```

## Run the Full Pipeline

```bash
python src/01_get_data.py
python src/02_train_tokenizer.py
python src/03_positional_encoding.py
python src/04_grouped_query_attention.py
python src/05_causal_masking.py
python src/06_moe_feed_forward.py
python src/07_norm_and_residuals.py
python src/08_transformer_block.py
```

Train:
```bash
python src/09_train_model.py --epochs 3 --batch-size 8 --block-size 128
```

Generate:
```bash
python src/10_generate_text.py --prompt "Once upon a time" --max-new-tokens 120
```

Test:
```bash
pytest -q
```

## Notes

This is a learning-focused small language model, not a production LLM. Output quality depends on model size, training duration, compute, and corpus quality.
