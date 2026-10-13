"""
generate.py: Step 6 - Autoregressive Story Generation and Sampling Strategies.

Decoding Strategies Explained:
1. Greedy Decoding (temperature -> 0):
   - Always picks the single highest probability token: argmax P(token).
   - Problem: Causes repetitive loops ("The cat saw the cat saw the cat...").
2. Temperature Scaling (T):
   - Divides logits by T before applying Softmax: Softmax(z / T).
   - T < 1.0: Sharper distribution, more confident and conservative.
   - T = 1.0: Normal sampling from model's predicted distribution.
   - T > 1.0: Flatter distribution, more surprising, chaotic, or creative.
3. Top-K Sampling:
   - Restricts sampling to only the K most likely tokens.
   - Prevents the model from ever choosing rare, nonsensical tokens from the long tail.
4. Top-P (Nucleus) Sampling:
   - Chooses the smallest set of tokens whose cumulative probability exceeds P (e.g. 0.90).
   - Dynamically expands candidate pool when multiple tokens are plausible,
     and narrows it to 1 or 2 tokens when the next word is obvious.
5. Repetition Penalty:
   - Discounts logits of previously generated tokens to break loop cycles.
"""

import os
import argparse
import torch
import torch.nn.functional as F
from model import TinyStoriesGPT, GPTConfig
from tokenizer_util import StoryTokenizer

def load_trained_model(checkpoint_path: str = "checkpoints/best_model.pt", device: str = None):
    """
    Loads model architecture and trained weights from checkpoint.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' not found! Run train.py first.")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = TinyStoriesGPT(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, config, checkpoint.get("val_loss", None)

@torch.no_grad()
def generate_story(
    prompt: str,
    checkpoint_path: str = "checkpoints/best_model.pt",
    tokenizer_dir: str = "tokenizer_data",
    max_new_tokens: int = 120,
    temperature: float = 0.8,
    top_k: int = 30,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    device: str = None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = StoryTokenizer(tokenizer_dir)
    model, config, val_loss = load_trained_model(checkpoint_path, device=device)

    # Encode prompt
    encoded_prompt = tokenizer.encode(prompt.strip())
    if len(encoded_prompt) == 0:
        encoded_prompt = [tokenizer.tokenizer.token_to_id("Once")]
    
    idx = torch.tensor([encoded_prompt], dtype=torch.long, device=device)

    print("\n" + "=" * 50)
    print(f"Prompt: \"{prompt}\"")
    print(f"Settings: Temp={temperature}, Top-K={top_k}, Top-P={top_p}, RepPenalty={repetition_penalty}")
    print("=" * 50 + "\n")

    for _ in range(int(max_new_tokens)):
        idx_cond = idx if idx.size(1) <= config.block_size else idx[:, -config.block_size:]

        logits, _ = model(idx_cond)
        logits = logits[:, -1, :].clone()

        # Repetition Penalty
        if repetition_penalty != 1.0:
            for token_id in set(idx[0].tolist()):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

        # Temperature
        logits = logits / max(temperature, 1e-5)

        # Top-K
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(int(top_k), logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        # Top-P
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

        # Check for EOS
        decoded = tokenizer.decode(idx[0].tolist())
        if "<|endoftext|>" in decoded:
            clean_story = decoded.split("<|endoftext|>")[0].strip()
            print(clean_story + "\n\n[End of Story]")
            print("\n" + "=" * 50 + "\n")
            return clean_story

    full_story = tokenizer.decode(idx[0].tolist()).replace("<|endoftext|>", "\n\n[End of Story]")
    print(full_story)
    print("\n" + "=" * 50 + "\n")
    return full_story

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate stories using TinyStories Mini-GPT")
    parser.add_argument("prompt", type=str, nargs="?", default="Once upon a time, there was a little boy named Timmy", help="Story opening prompt")
    parser.add_argument("--max_tokens", type=int, default=120, help="Maximum new tokens to generate")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=30, help="Top-K sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P nucleus sampling")
    parser.add_argument("--rep_penalty", type=float, default=1.15, help="Repetition penalty")
    args = parser.parse_args()

    try:
        generate_story(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.rep_penalty
        )
    except FileNotFoundError as e:
        print(f"[!] {e}")
