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
   - Dynamically expands the candidate pool when multiple tokens are plausible,
     and narrows it to 1 or 2 tokens when the next word is obvious.
"""

import os
import torch
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

def generate_story(
    prompt: str,
    checkpoint_path: str = "checkpoints/best_model.pt",
    tokenizer_dir: str = "tokenizer_data",
    max_new_tokens: int = 150,
    temperature: float = 0.8,
    top_k: int = 30,
    top_p: float = 0.9,
    device: str = None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = StoryTokenizer(tokenizer_dir)
    model, config, val_loss = load_trained_model(checkpoint_path, device=device)

    # Encode prompt
    encoded_prompt = tokenizer.encode(prompt)
    if len(encoded_prompt) == 0:
        encoded_prompt = [tokenizer.tokenizer.token_to_id("Once")]
    
    input_tensor = torch.tensor([encoded_prompt], dtype=torch.long, device=device)

    print(f"\n==================================================")
    print(f"Prompt: \"{prompt}\"")
    print(f"Sampling: Temp={temperature}, Top-K={top_k}, Top-P={top_p}")
    print(f"==================================================")

    generated_ids = model.generate(
        idx=input_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_id=tokenizer.eos_token_id
    )

    story = tokenizer.decode(generated_ids[0].tolist())
    # Clean up end-of-text token if present
    story = story.replace("<|endoftext|>", "\n[End of Story]")
    print(f"\n{story}\n")
    print(f"==================================================\n")
    return story

if __name__ == "__main__":
    import sys
    prompt = "Once upon a time, there was a little boy named Timmy"
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    
    try:
        generate_story(prompt)
    except FileNotFoundError as e:
        print(f"[!] {e}")
