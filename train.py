"""
train.py: Step 5 - Training Loop, Optimization, and Learning Rate Scheduling.

Key Techniques:
1. AdamW with Decoupled Weight Decay:
   - Weight decay regularizes large weights to prevent overfitting.
   - We apply weight decay ONLY to 2D matrix weights, and NOT to 1D biases or LayerNorm scales.
2. Learning Rate Warmup & Cosine Decay:
   - Starting immediately with a high LR destabilizes random initial weights.
   - Warmup ramps LR up linearly for the first few hundred steps.
   - Cosine decay gradually drops LR down to ~10% of maximum to help fine-tune representations.
3. Gradient Clipping:
   - Caps gradient norm at 1.0 to prevent sudden gradient explosion from destabilizing training.
4. Live Generation Checks:
   - Periodically prompts the model during training to watch text generation improve in real time!
"""

import os
import math
import time
import torch
import torch.nn as nn
from torch.optim import AdamW

from model import TinyStoriesGPT, GPTConfig
from tokenizer_util import StoryTokenizer, train_custom_bpe
from dataset import StoryTextDataset, get_dataloader

def get_lr(it: int, warmup_iters: int, max_iters: int, lr: float, min_lr: float) -> float:
    """
    Cosine learning rate schedule with linear warmup.
    """
    # 1. Linear warmup
    if it < warmup_iters:
        return lr * (it + 1) / warmup_iters
    # 2. If beyond max_iters, return min_lr
    if it > max_iters:
        return min_lr
    # 3. In between, use cosine decay down to min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)

def configure_optimizers(model: TinyStoriesGPT, weight_decay: float, learning_rate: float, betas=(0.9, 0.95)):
    """
    Separates parameters into those that receive weight decay (2D weights)
    and those that do not (1D biases, LayerNorm parameters).
    """
    decay_params = []
    nodecay_params = []
    for pn, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay_params.append(p)
        else:
            nodecay_params.append(p)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = AdamW(optim_groups, lr=learning_rate, betas=betas)
    return optimizer

@torch.no_grad()
def estimate_loss(model: TinyStoriesGPT, val_loader, eval_iters: int = 20, device: str = "cpu") -> float:
    """
    Estimates average validation loss across several batches.
    """
    model.eval()
    losses = []
    iter_val = iter(val_loader)
    for _ in range(eval_iters):
        try:
            x, y = next(iter_val)
        except StopIteration:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))

def train(
    train_file: str = "data/train.txt",
    val_file: str = "data/val.txt",
    tokenizer_dir: str = "tokenizer_data",
    checkpoint_dir: str = "checkpoints",
    max_iters: int = 1000,
    batch_size: int = 16,
    block_size: int = 128,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 128,
    lr: float = 1e-3,
    min_lr: float = 1e-4,
    warmup_iters: int = 100,
    eval_interval: int = 200,
    eval_iters: int = 20,
    device: str = None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Training on device: {device.upper()}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Initialize tokenizer
    if not os.path.exists(os.path.join(tokenizer_dir, "vocab.json")):
        print("[*] Tokenizer not found. Training BPE tokenizer first...")
        train_custom_bpe(data_file=train_file, vocab_size=4096, output_dir=tokenizer_dir)
    
    tokenizer = StoryTokenizer(tokenizer_dir)
    print(f"[+] Tokenizer ready! Vocab size: {tokenizer.vocab_size}")

    # 2. Build datasets and data loaders
    train_loader = get_dataloader(train_file, tokenizer, block_size=block_size, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_file, tokenizer, block_size=block_size, batch_size=batch_size, shuffle=False)

    # 3. Initialize Model
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=0.1
    )
    model = TinyStoriesGPT(config).to(device)
    print(f"[+] Model instantiated: {model.get_num_params():,} parameters")

    # 4. Setup Optimizer
    optimizer = configure_optimizers(model, weight_decay=1e-1, learning_rate=lr)

    # 5. Training Loop
    model.train()
    train_iter = iter(train_loader)
    best_val_loss = float("inf")
    start_time = time.time()

    print(f"\n[*] Commencing training for {max_iters} iterations...")
    print(f"    Batch Size: {batch_size} | Block Size: {block_size} | Tokens/Batch: {batch_size * block_size:,}")

    for it in range(1, max_iters + 1):
        # Update learning rate according to warmup + cosine decay
        current_lr = get_lr(it, warmup_iters, max_iters, lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        # Fetch next batch
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        # Forward pass & loss calculation
        logits, loss = model(x, y)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()

        # Logging & evaluation
        if it % 50 == 0 or it == 1:
            tokens_per_sec = (it * batch_size * block_size) / (time.time() - start_time)
            print(f"Iter {it:4d}/{max_iters} | Train Loss: {loss.item():.4f} | LR: {current_lr:.6f} | Speed: {tokens_per_sec:.0f} tokens/s")

        if it % eval_interval == 0 or it == max_iters:
            val_loss = estimate_loss(model, val_loader, eval_iters=eval_iters, device=device)
            perplexity = math.exp(min(val_loss, 20.0))
            print(f"\n>>> [EVALUATION @ Iter {it}] Val Loss: {val_loss:.4f} | Perplexity: {perplexity:.2f}")

            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
                torch.save({"config": config, "state_dict": model.state_dict(), "iter": it, "val_loss": val_loss}, ckpt_path)
                print(f"    Saved best checkpoint to {ckpt_path}")

            # Live text generation preview!
            prompt = "Once upon a time, Lily"
            input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
            gen_ids = model.generate(input_ids, max_new_tokens=40, temperature=0.8, top_k=20, eos_id=tokenizer.eos_token_id)
            generated_text = tokenizer.decode(gen_ids[0].tolist())
            print(f"    Sample Generation:\n    \"{generated_text}\"\n")

    print(f"\n[+] Training completed in {time.time() - start_time:.1f}s. Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train TinyStories GPT Model")
    parser.add_argument("--max_iters", type=int, default=400, help="Maximum training iterations")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--block_size", type=int, default=128, help="Context length")
    parser.add_argument("--n_layer", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--eval_interval", type=int, default=100, help="Evaluation interval")
    args = parser.parse_args()

    train(
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        lr=args.lr,
        eval_interval=args.eval_interval
    )
