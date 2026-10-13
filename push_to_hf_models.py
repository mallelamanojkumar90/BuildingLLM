"""
push_to_hf_models.py: Publish TinyStories Mini-GPT to the Hugging Face Model Hub.

This script:
1. Creates a model repository on Hugging Face (e.g., https://huggingface.co/mallelamanoj75/tinystories-minigpt).
2. Generates a metadata-rich Model Card (README.md) and config.json.
3. Bundles the model weights, tokenizer files, and architecture code.
4. Uploads everything directly to the Model Hub.
"""

import os
import json
import shutil
import argparse
from huggingface_hub import HfApi, create_repo

MODEL_CARD_TEMPLATE = """---
language:
- en
license: mit
tags:
- text-generation
- pytorch
- gpt
- tinystories
- causal-lm
datasets:
- roneneldan/TinyStories
pipeline_tag: text-generation
---

# 📖 TinyStories Mini-GPT (1.3M Parameters)

A compact, decoder-only autoregressive language model built from scratch in pure PyTorch and trained on Microsoft Research's **TinyStories** dataset.

## 🔍 Model Overview
- **Architecture:** Decoder-Only Transformer (Pre-LayerNorm, Causal Multi-Head Attention)
- **Parameters:** 1,329,408 (~1.33 Million)
- **Layers:** 4 Transformer Blocks
- **Attention Heads:** 4 heads ($d_k = 32$)
- **Hidden Dimension:** $d_{model} = 128$
- **Vocabulary:** 4,096 tokens (Byte-Level BPE)
- **Context Length:** 128 tokens
- **Training Corpus:** [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)

---

## 🚀 How to Use

### Quick Inference via Python:
```python
import os
import torch
from huggingface_hub import hf_hub_download

# 1. Download model files from Hugging Face
repo_id = "{repo_id}"
ckpt_path = hf_hub_download(repo_id=repo_id, filename="best_model.pt")
vocab_path = hf_hub_download(repo_id=repo_id, filename="vocab.json")
merges_path = hf_hub_download(repo_id=repo_id, filename="merges.txt")
model_py = hf_hub_download(repo_id=repo_id, filename="model.py")
tokenizer_py = hf_hub_download(repo_id=repo_id, filename="tokenizer_util.py")

# 2. Load model
from model import TinyStoriesGPT, GPTConfig
from tokenizer_util import StoryTokenizer

os.makedirs("tokenizer_data", exist_ok=True)
import shutil
shutil.copy(vocab_path, "tokenizer_data/vocab.json")
shutil.copy(merges_path, "tokenizer_data/merges.txt")

tok = StoryTokenizer("tokenizer_data")
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model = TinyStoriesGPT(checkpoint["config"])
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# 3. Generate a story!
prompt = "Once upon a time, Lily found a sparkling green stone"
input_ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
output_ids = model.generate(input_ids, max_new_tokens=80, temperature=0.8, top_k=30, top_p=0.9)
print(tok.decode(output_ids[0].tolist()))
```

---

## 📊 Sample Generation
> **Prompt:** *"One sunny morning, Mia and her mom went to the big park"*  
>  
> *"One sunny morning, Mia and her mom went to the big park with her mom. They saw a big, red ball and a swing.*  
> *"Look, this is a big slide!" Mia said. "But I want to swing very high and fast."*  
> *"No, I want to play!" Mia said. "Let's play!"*  
> *They ran to the swing and jumped in the slide. They were happy. They had fun.*  
> *[End of Story]"*

---

## 📚 Citation & References
- [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759) (Eldan & Li, 2023)
- GitHub Repository: [mallelamanojkumar90/BuildingLLM](https://github.com/mallelamanojkumar90/BuildingLLM)
"""

def prepare_model_bundle(bundle_dir: str = "hf_model_dist", repo_id: str = ""):
    """
    Assembles model weights, tokenizer, config, and documentation into a bundle directory.
    """
    if os.path.exists(bundle_dir):
        shutil.rmtree(bundle_dir)
    os.makedirs(bundle_dir, exist_ok=True)

    print(f"[*] Packaging model bundle into '{bundle_dir}'...")

    # 1. Copy model weights
    ckpt_src = os.path.join("checkpoints", "best_model.pt")
    if not os.path.exists(ckpt_src):
        raise FileNotFoundError(f"Missing '{ckpt_src}'. Please train the model first.")
    shutil.copy(ckpt_src, os.path.join(bundle_dir, "best_model.pt"))

    # 2. Copy code files for easy standalone import
    shutil.copy("model.py", os.path.join(bundle_dir, "model.py"))
    shutil.copy("tokenizer_util.py", os.path.join(bundle_dir, "tokenizer_util.py"))
    shutil.copy("generate.py", os.path.join(bundle_dir, "generate.py"))

    # 3. Copy tokenizer files directly to the root for Model Hub compatibility
    shutil.copy(os.path.join("tokenizer_data", "vocab.json"), os.path.join(bundle_dir, "vocab.json"))
    shutil.copy(os.path.join("tokenizer_data", "merges.txt"), os.path.join(bundle_dir, "merges.txt"))

    # 4. Generate config.json
    config_dict = {
        "model_type": "tinystories-gpt",
        "vocab_size": 4096,
        "block_size": 128,
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 128,
        "dropout": 0.1,
        "bias": False,
        "total_params": 1329408,
        "dataset": "roneneldan/TinyStories",
        "task": "causal-lm"
    }
    with open(os.path.join(bundle_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # 5. Write Model Card README.md
    with open(os.path.join(bundle_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(MODEL_CARD_TEMPLATE.replace("{repo_id}", repo_id))

    print(f"[+] Model bundle prepared successfully.")

def push_model_to_hf(model_name: str = "tinystories-minigpt", private: bool = False):
    api = HfApi()
    user_info = api.whoami()
    username = user_info["name"]
    repo_id = f"{username}/{model_name}"
    bundle_dir = "hf_model_dist"

    print("\n==================================================")
    print(f"[*] Hugging Face Account: {username}")
    print(f"[*] Destination Model:    {repo_id}")
    print("==================================================")

    # 1. Create Model Repository on Hugging Face Model Hub
    print(f"[*] Creating model repository '{repo_id}' on Hugging Face...")
    url = create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True
    )
    print(f"[+] Model repository ready at: {url}")

    # 2. Prepare files to upload
    prepare_model_bundle(bundle_dir, repo_id=repo_id)

    # 3. Upload folder to Hugging Face Model Hub
    print(f"[*] Uploading model weights, tokenizer, and config to Hugging Face...")
    api.upload_folder(
        folder_path=bundle_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload TinyStories Mini-GPT weights, tokenizer, and model card"
    )

    # 4. Cleanup local staging bundle
    shutil.rmtree(bundle_dir)

    print("\n[+] Model successfully published to Hugging Face Models!")
    print(f"[*] Live Model Hub URL: https://huggingface.co/{repo_id}")
    print("==================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push Mini-GPT model to Hugging Face Models")
    parser.add_argument("--model_name", type=str, default="tinystories-minigpt", help="Model repo name")
    parser.add_argument("--private", action="store_true", help="Make model private")
    args = parser.parse_args()

    push_model_to_hf(model_name=args.model_name, private=args.private)
