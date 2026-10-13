"""
data_prep.py: Step 1 - Downloading and Preparing the Short Stories Dataset.

Why TinyStories?
- Most internet datasets (Common Crawl, Wikipedia, Books3) are hundreds of gigabytes and
  contain complex, adult-level vocabulary, which requires models with billions of parameters.
- TinyStories (Eldan & Li, 2023) contains synthetic short stories generated with vocabulary
  typically understood by 3 to 4-year-olds.
- It enables small language models (<10M parameters) to learn English syntax, coherent reasoning,
  and narrative arcs without requiring a supercomputer.
"""

import os
from datasets import load_dataset
from tqdm import tqdm

def download_and_save_tinystories(
    output_dir: str = "data",
    num_train_stories: int = 10000,
    num_val_stories: int = 1000
):
    """
    Streams a manageable subset of TinyStories from Hugging Face and saves them as raw text files.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.txt")
    val_path = os.path.join(output_dir, "val.txt")

    print(f"[*] Streaming TinyStories from Hugging Face...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    print(f"[*] Collecting {num_train_stories} training stories...")
    train_count = 0
    val_count = 0
    
    with open(train_path, "w", encoding="utf-8") as f_train, \
         open(val_path, "w", encoding="utf-8") as f_val:
        
        iterator = iter(ds)
        
        # 1. Collect training stories
        pbar_train = tqdm(total=num_train_stories, desc="Saving train stories")
        while train_count < num_train_stories:
            story = next(iterator)["text"].strip()
            if len(story) > 50:  # Skip degenerate/empty samples
                f_train.write(story + "\n<|endoftext|>\n")
                train_count += 1
                pbar_train.update(1)
        pbar_train.close()

        # 2. Collect validation stories
        pbar_val = tqdm(total=num_val_stories, desc="Saving val stories")
        while val_count < num_val_stories:
            story = next(iterator)["text"].strip()
            if len(story) > 50:
                f_val.write(story + "\n<|endoftext|>\n")
                val_count += 1
                pbar_val.update(1)
        pbar_val.close()

    train_size_mb = os.path.getsize(train_path) / (1024 * 1024)
    val_size_mb = os.path.getsize(val_path) / (1024 * 1024)
    print(f"[+] Done! Dataset saved successfully:")
    print(f"    Train: {train_path} ({train_count} stories, {train_size_mb:.2f} MB)")
    print(f"    Val:   {val_path} ({val_count} stories, {val_size_mb:.2f} MB)")

if __name__ == "__main__":
    download_and_save_tinystories(
        output_dir="data",
        num_train_stories=8000,
        num_val_stories=800
    )
