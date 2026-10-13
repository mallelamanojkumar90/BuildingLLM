# 🧠 Building a Small Language Model (Mini-GPT) from Scratch

A complete, self-contained implementation of an autoregressive Decoder-Only Transformer (Mini-GPT) built in pure PyTorch and trained from scratch on Microsoft Research's **TinyStories** dataset.

This repository demonstrates every step of modern LLM pretraining: streaming internet datasets, custom subword Byte-Pair Encoding (BPE), causal multi-head self-attention, learning rate scheduling with warmup and cosine decay, and nucleus (top-p) sampling.

---

## 📑 Table of Contents
- [Architecture Overview](#-architecture-overview)
- [Why TinyStories?](#-why-tinystories)
- [Repository Structure](#-repository-structure)
- [Key Concepts & Architectural Decisions](#-key-concepts--architectural-decisions)
- [Quickstart Guide](#-quickstart-guide)
  - [1. Installation](#1-installation)
  - [2. Download & Prepare Corpus](#2-download--prepare-corpus)
  - [3. Train Custom BPE Tokenizer](#3-train-custom-bpe-tokenizer)
  - [4. Train the Model](#4-train-the-model)
  - [5. Generate Stories](#5-generate-stories)
- [Hyperparameters & Model Size](#-hyperparameters--model-size)
- [Training Progress & Live Results](#-training-progress--live-results)
- [Decoding & Sampling Strategies](#-decoding--sampling-strategies)
- [Engineering Challenges Addressed](#-engineering-challenges-addressed)
- [Acknowledgments](#-acknowledgments)

---

## 🏗️ Architecture Overview

The model follows a modern Pre-LayerNorm (Pre-LN) Decoder-Only Transformer architecture (similar to GPT-2, LLaMA, and Mistral):

```text
Input Tokens (B, T)
        │
        ▼
Token Embeddings (V × d) + Positional Embeddings (T × d)
        │
        ├──► [Transformer Block 1] ───┐
        │    - Pre-LayerNorm          │
        │    - Causal Self-Attention  │ Repeated N times (n_layer = 4)
        │    - Residual Connection    │
        │    - Pre-LayerNorm          │
        │    - FeedForward MLP (GELU) │
        │    - Residual Connection ───┘
        │
        ├──► [Transformer Block N]
        │
        ▼
Final LayerNorm (d)
        │
        ▼
Language Model Head (Linear projection to V)
        │
        ▼
Next-Token Logits (B, T, V)
        │
        ▼
Autoregressive Sampling (Temperature / Top-K / Top-P)
```

---

## 📖 Why TinyStories?

Standard web datasets (Common Crawl, Wikipedia, C4) contain complex vocabularies and multi-domain nuances requiring billions of parameters and GPU clusters to produce coherent English sentences.

**TinyStories** ([Eldan & Li, 2023](https://arxiv.org/abs/2305.07759)) consists of synthetic short stories generated with vocabulary understood by 3- to 4-year-olds:
- Preserves natural grammar, dialogue, reasoning, and consistent narrative arcs.
- Allows small models with **~1M to 10M parameters** to quickly learn fluent English and coherent storytelling.
- Runs and trains efficiently even on a **standard CPU** in minutes.

---

## 📁 Repository Structure

```text
BuildingLLM/
├── data_prep.py         # Step 1: Stream & clean stories from Hugging Face
├── tokenizer_util.py    # Step 2: Custom Byte-Level BPE tokenizer (vocab: 4096)
├── dataset.py           # Step 3: PyTorch Dataset & shifted context windows (x -> y)
├── model.py             # Step 4: Pure PyTorch Decoder-Only Transformer
├── train.py             # Step 5: AdamW training loop, cosine LR scheduler, clipping
├── generate.py          # Step 6: Text generation with Temperature, Top-K, Top-P
├── requirements.txt     # Python package dependencies
├── .gitignore           # Ignores large datasets, caches, and checkpoints
├── README.md            # Comprehensive project documentation
├── data/                # Data directory (populated via data_prep.py)
├── tokenizer_data/      # Trained BPE tokenizer files (vocab.json, merges.txt)
└── checkpoints/         # Saved model weights (populated via train.py)
```

---

## 💡 Key Concepts & Architectural Decisions

1. **Byte-Level Byte-Pair Encoding (BPE):**
   - Standard GPT-2 tokenizers use a vocabulary of 50,257 tokens. For a compact model ($d_{model} = 128$), an embedding matrix of 50k tokens requires over 6.4 million parameters just for the embedding layer!
   - We train a custom Byte-Level BPE tokenizer of **4,096 tokens**, keeping the embedding layer to only ~524k parameters while guaranteeing zero out-of-vocabulary (`<unk>`) errors.

2. **Causal Attention Masking:**
   - To enforce the autoregressive property (predicting token $t$ using only tokens $1 \dots t-1$), attention scores for future positions are masked with $-\infty$ before applying softmax ($e^{-\infty} = 0$).

3. **Pre-LayerNorm (Pre-LN):**
   - LayerNorm is placed before each multi-head attention and MLP block. This keeps the residual stream as an identity highway for unimpeded gradient flow, eliminating vanishing/exploding gradients.

4. **Weight Tying:**
   - The token embedding table weights (`wte.weight`) and the final projection head (`lm_head.weight`) are tied, saving memory and aligning representations.

5. **GELU Non-Linearity:**
   - Uses Gaussian Error Linear Units instead of standard ReLU for smooth, non-zero gradients across small negative activations.

---

## 🚀 Quickstart Guide

### 1. Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/mallelamanojkumar90/BuildingLLM.git
cd BuildingLLM
pip install -r requirements.txt
```

### 2. Download & Prepare Corpus
Streams 8,000 training stories and 800 validation stories from Hugging Face:
```bash
python data_prep.py
```

### 3. Train Custom BPE Tokenizer
Trains a 4,096-token Byte-Level BPE tokenizer:
```bash
python tokenizer_util.py
```

### 4. Train the Model
Launch training with AdamW, cosine decay, and live generation previews:
```bash
# Fast CPU run (~2.5 minutes, 300 iterations)
python train.py --max_iters 300 --eval_interval 100 --batch_size 16

# Extended training for richer coherence (1000 - 2000 iterations)
python train.py --max_iters 1500 --eval_interval 200 --batch_size 32
```

### 5. Generate Stories
Prompt the trained model using interactive inference:
```bash
python generate.py "Once upon a time, Timmy found a magical key"
```

---

## ⚙️ Hyperparameters & Model Size

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Vocabulary Size ($V$)** | 4,096 | Compact BPE representation avoiding embedding explosion |
| **Context Length ($T$)** | 128 | Story context window (block size) |
| **Transformer Blocks ($N$)** | 4 | Depth sufficient for syntax and narrative patterns |
| **Attention Heads ($h$)** | 4 | Multi-head attention ($d_k = 128 / 4 = 32$) |
| **Embedding Dimension ($d$)** | 128 | Hidden state width |
| **Dropout** | 0.1 | Regularization against overfitting |
| **Total Parameters** | **1,329,408** | Fast training on CPU (~3,500 tokens/sec) |

---

## 📊 Training Progress & Live Results

During pretraining, loss progression follows theoretical expectations:
- **Initialization (Random Chance):** Initial cross-entropy loss is:
  $$\mathcal{L}_{\text{init}} \approx \ln(4096) \approx 8.3178$$
  Our model initialized at **8.3295**, validating correct Gaussian weight initialization.
- **Iteration 100:** Train Loss drops to `5.2178` (Perplexity: 201.26).
- **Iteration 200:** Train Loss drops to `4.2351` (Perplexity: 97.63).
- **Iteration 300:** Train Loss reaches `4.3274` / Val Loss `4.4228` (Perplexity: 83.33).

### Sample Output Evolution:
* **Step 100:**
  > *"Once upon a time, Lily,. The was a girl her with a saw to the park and was the her mom in the little day..."*
* **Step 300:**
  > *"Once upon a time, Timmy found a little puppy. Lily asked the park... Tom and said, 'Yes, a small park.' She said, 'I not like to play with the tree.' The boy named Timmy was a little girl smiled. He asked his little girl. She went to help. She asked, 'I'm proud of the park!' [End of Story]"*

Notice how the model quickly acquires quotation marks, character names, grammatical transitions, and termination tokens.

---

## 🎛️ Decoding & Sampling Strategies

The generation script ([`generate.py`](file:///c:/Manojkumar/development/BuildingLLM/generate.py)) supports three complementary decoding techniques:

1. **Temperature Scaling ($T$):**
   Divides logits by $T$ prior to softmax:
   - $T < 0.7$: Conservative, highly focused, and repetitive.
   - $T = 0.8$: Balanced between creativity and grammatical structure.
   - $T > 1.1$: High variance and adventurous vocabulary.
2. **Top-K Filtering:** Restricts candidate selection to the $K$ highest-probability tokens, removing low-probability long-tail words.
3. **Top-P (Nucleus) Filtering:** Dynamically restricts candidates to the smallest subset whose cumulative probability reaches $P$ (e.g., $0.90$).

---

## 🛡️ Engineering Challenges Addressed

1. **Embedding Matrix Memory Bottleneck:** Solved by training a compact subword tokenizer (4,096 tokens) rather than standard 50k+ GPT-2 vocabs, and tying weights between `wte` and `lm_head`.
2. **Causal Information Leakage:** Solved with a strict lower-triangular causal attention mask.
3. **Training Instability:** Solved with Pre-LayerNorm residual pathways, linear LR warmup (100 steps), cosine decay, and gradient clipping at norm 1.0.
4. **Repetition & Mode Collapse:** Solved by pairing Top-P nucleus sampling and Top-K filtering rather than greedy `argmax` search.

---

## 📚 Acknowledgments

- [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759) (Ronen Eldan & Yuanzhi Li, Microsoft Research, 2023)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy for clean Transformer design patterns.
