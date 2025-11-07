 Gutenberg Text Transformer 🧠

A from-scratch implementation of a Transformer-based language model trained on text from Project Gutenberg.  
This project is part of my hands-on journey to understand how Transformers (like GPT) work at the architecture level.

---

 🧭 Overview
The goal of this project is to build a miniature GPT-style model step-by-step — from raw text to a trained Transformer that can generate sentences.  
Each step adds one major concept, focusing on clarity and code transparency over scale.

---

 🪜 10-Step Roadmap

1. Get the Data – Download and clean text datasets from Project Gutenberg using Python’s `requests`.  
2. Train the Tokenizer – Implement a Byte-Pair Encoding (BPE) tokenizer to convert text into subword tokens.  
3. Positional Encoding – Add Rotary Positional Encoding (RoPE) so the model understands word order.  
4. Grouped Query Attention (GQA) – Implement the attention mechanism that powers all modern Transformers.  
5. Causal Masking – Ensure the model only “looks backward” during training, not ahead at future tokens.  
6. Feed-Forward (MoE) Layer – Add a Mixture-of-Experts (MoE) layer for efficient non-linear transformations.  
7. Normalization & Skip Connections – Stabilize training with RMSNorm and residual links.  
8. Full Transformer Block – Stack all modules (Attention + MLP + Norm) into a complete decoder block.  
9. Training Loop – Train the model with PyTorch using self-supervised next-token prediction.  
10. Text Generation – Generate text using sampling or nucleus decoding, and visualize results.

---

 ✅ Progress So Far

- ✅ Step 1: Downloaded and cleaned dataset from Project Gutenberg  
- ✅ Step 2: Trained BPE tokenizer and saved `gutenberg_tokenizer.json`  
- ✅ Step 3: Implemented Rotary Positional Encoding  
- ⬜ Step 4: Building Grouped Query Attention module  
- ⬜ Step 5–10: Coming soon 🚀

