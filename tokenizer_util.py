"""
tokenizer_util.py: Step 2 - Subword Tokenization (Byte-Level BPE).

Why Byte-Pair Encoding (BPE)?
1. Character-level Tokenization:
   - Vocab is tiny (~100 chars), but sequence length is huge.
   - Since Transformer attention is O(N^2) in sequence length, long sequences are expensive.
2. Word-level Tokenization:
   - Vocab explodes (100,000+ words). Rare words or typos become <UNK> (unknown).
3. Byte-Level BPE (Used by GPT-2, GPT-4, LLaMA):
   - Starts at the raw byte level (256 base bytes).
   - Merges the most frequent byte pairs iteratively up to a chosen vocabulary size (e.g., 4096).
   - Result: Common words like "the", "playing", "forest" become single tokens.
   - Rare words or typos break down into smaller subwords or raw bytes. Zero out-of-vocabulary tokens!
   - Keeps vocabulary compact so the embedding table (V x d_model) does not consume all RAM/CPU.
"""

import os
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

TOKENIZER_DIR = "tokenizer_data"
SPECIAL_TOKENS = ["<|pad|>", "<|endoftext|>"]

def train_custom_bpe(
    data_file: str = "data/train.txt",
    vocab_size: int = 4096,
    min_frequency: int = 2,
    output_dir: str = TOKENIZER_DIR
):
    """
    Trains a Byte-Level BPE tokenizer on our stories dataset and saves it to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Training Byte-Level BPE tokenizer on {data_file} (Vocab size: {vocab_size})...")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[data_file],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS
    )

    tokenizer.save_model(output_dir)
    print(f"[+] Tokenizer saved to {output_dir}/")
    return tokenizer

def load_custom_tokenizer(output_dir: str = TOKENIZER_DIR):
    """
    Loads the trained Byte-Level BPE tokenizer.
    """
    vocab_file = os.path.join(output_dir, "vocab.json")
    merges_file = os.path.join(output_dir, "merges.txt")
    if not os.path.exists(vocab_file) or not os.path.exists(merges_file):
        raise FileNotFoundError(
            f"Tokenizer files not found in '{output_dir}'. Please train the tokenizer first!"
        )

    tokenizer = ByteLevelBPETokenizer(vocab_file, merges_file)
    return tokenizer

class StoryTokenizer:
    """
    Convenience wrapper providing simple encode / decode methods and token IDs.
    """
    def __init__(self, tokenizer_dir: str = TOKENIZER_DIR):
        self.tokenizer = load_custom_tokenizer(tokenizer_dir)
        self.pad_token_id = self.tokenizer.token_to_id("<|pad|>")
        self.eos_token_id = self.tokenizer.token_to_id("<|endoftext|>")
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

if __name__ == "__main__":
    # Test training the tokenizer if data exists
    if os.path.exists("data/train.txt"):
        train_custom_bpe()
        tok = StoryTokenizer()
        sample = "Once upon a time, Lily found a sparkling green stone in the garden."
        encoded = tok.encode(sample)
        decoded = tok.decode(encoded)
        print(f"Original: {sample}")
        print(f"Tokens:   {encoded} (Count: {len(encoded)})")
        print(f"Decoded:  {decoded}")
        print(f"Vocab size: {tok.vocab_size}")
    else:
        print("[!] data/train.txt not found. Run data_prep.py first.")
