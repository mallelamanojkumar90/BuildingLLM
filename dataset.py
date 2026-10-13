"""
dataset.py: Step 3 - Data Pipeline, Context Windows, and Target Shifting.

How does an LLM learn to predict the next word?
Language modeling is framed as self-supervised next-token prediction:
If sequence = [t0, t1, t2, t3, t4, t5] and context length T = 4:
    Input  x: [t0, t1, t2, t3]
    Target y: [t1, t2, t3, t4]

Why this is efficient:
In a single forward pass, the model makes T separate predictions simultaneously:
- Given [t0], it predicts t1
- Given [t0, t1], it predicts t2
- Given [t0, t1, t2], it predicts t3
- Given [t0, t1, t2, t3], it predicts t4
Thanks to Causal Masking, none of the earlier positions can see the later positions.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer_util import StoryTokenizer

class StoryTextDataset(Dataset):
    """
    Dataset that loads tokenized stories into a flat 1D tensor and slices contiguous chunks of length (block_size + 1).
    """
    def __init__(self, text_file: str, tokenizer: StoryTokenizer, block_size: int = 128):
        self.block_size = block_size
        
        if not os.path.exists(text_file):
            raise FileNotFoundError(f"File {text_file} does not exist. Run data_prep.py first.")

        print(f"[*] Reading and tokenizing {text_file}...")
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read()

        token_ids = tokenizer.encode(text)
        self.data = torch.tensor(token_ids, dtype=torch.long)
        print(f"[+] Loaded {len(self.data):,} tokens from {text_file}.")
        print(f"    Possible training sequences of length {block_size}: {len(self):,}")

    def __len__(self):
        # We need block_size + 1 tokens: block_size for x, and 1 for the shifted target y
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx: int):
        # Slice block_size + 1 tokens
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1] # Inputs:  tokens 0 to T-1
        y = chunk[1:]  # Targets: tokens 1 to T (shifted by 1)
        return x, y

def get_dataloader(
    text_file: str,
    tokenizer: StoryTokenizer,
    block_size: int = 128,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Creates an optimized PyTorch DataLoader.
    """
    dataset = StoryTextDataset(text_file, tokenizer, block_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False, # Set to False for CPU
        drop_last=True
    )
    return dataloader

if __name__ == "__main__":
    if os.path.exists("tokenizer_data/vocab.json") and os.path.exists("data/train.txt"):
        tok = StoryTokenizer()
        loader = get_dataloader("data/train.txt", tok, block_size=16, batch_size=2)
        x_batch, y_batch = next(iter(loader))
        print("\n--- Batch Inspection ---")
        print("x_batch shape:", x_batch.shape)
        print("y_batch shape:", y_batch.shape)
        print("x_batch[0]:", x_batch[0].tolist())
        print("y_batch[0]:", y_batch[0].tolist())
        print("Decoded x:", tok.decode(x_batch[0].tolist()))
        print("Decoded y:", tok.decode(y_batch[0].tolist()))
    else:
        print("[!] Tokenizer or dataset not prepared yet.")
