"""
deploy_to_hf.py: Automated Deployment Script to Hugging Face Spaces.

This script:
1. Validates Hugging Face credentials.
2. Creates a dedicated staging directory containing only the files needed for inference.
3. Automatically creates a Gradio Space on Hugging Face (if it doesn't already exist).
4. Uploads all files directly to the Space.
"""

import os
import shutil
import argparse
from huggingface_hub import HfApi, create_repo

SPACE_README_TEMPLATE = """---
title: TinyStories Mini-GPT
emoji: 📖
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.2.0
app_file: app.py
pinned: false
license: mit
---

# 📖 TinyStories Mini-GPT

An interactive children's story generator powered by a **1.3M parameter Decoder-Only Transformer** built from scratch in pure PyTorch and trained on Microsoft Research's **TinyStories** dataset.

### 🔍 Model Architecture
- **Layers:** 4 Transformer Blocks
- **Heads:** 4 Causal Multi-Head Attention heads
- **Hidden Dimension:** $d_{model} = 128$
- **Vocabulary:** 4,096 tokens (Byte-Level BPE)
- **Context Length:** 128 tokens
"""

HF_REQUIREMENTS = """torch>=2.0.0
tokenizers>=0.15.0
gradio>=4.0.0
"""

def prepare_staging_directory(staging_dir: str = "hf_space_dist"):
    """
    Copies only the necessary files for the Hugging Face Space into a staging directory.
    """
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)

    print(f"[*] Preparing staging directory '{staging_dir}'...")

    # 1. Copy core inference code
    shutil.copy("app.py", os.path.join(staging_dir, "app.py"))
    shutil.copy("model.py", os.path.join(staging_dir, "model.py"))
    shutil.copy("tokenizer_util.py", os.path.join(staging_dir, "tokenizer_util.py"))

    # 2. Copy tokenizer data
    dest_tok = os.path.join(staging_dir, "tokenizer_data")
    os.makedirs(dest_tok, exist_ok=True)
    shutil.copy(os.path.join("tokenizer_data", "vocab.json"), os.path.join(dest_tok, "vocab.json"))
    shutil.copy(os.path.join("tokenizer_data", "merges.txt"), os.path.join(dest_tok, "merges.txt"))

    # 3. Copy model checkpoint
    dest_ckpt = os.path.join(staging_dir, "checkpoints")
    os.makedirs(dest_ckpt, exist_ok=True)
    checkpoint_src = os.path.join("checkpoints", "best_model.pt")
    if not os.path.exists(checkpoint_src):
        raise FileNotFoundError(f"Missing {checkpoint_src}. Please train the model before deploying!")
    shutil.copy(checkpoint_src, os.path.join(dest_ckpt, "best_model.pt"))

    # 4. Write requirements.txt
    with open(os.path.join(staging_dir, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write(HF_REQUIREMENTS)

    # 5. Write Space README.md with YAML metadata
    with open(os.path.join(staging_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(SPACE_README_TEMPLATE)

    print(f"[+] Staging directory ready with all deployment artifacts.")

def deploy(space_name: str = "tinystories-minigpt", private: bool = False):
    api = HfApi()
    user_info = api.whoami()
    username = user_info["name"]
    repo_id = f"{username}/{space_name}"
    staging_dir = "hf_space_dist"

    print(f"\n==================================================")
    print(f"[*] Authenticated as: {username}")
    print(f"[*] Target Space:     {repo_id}")
    print(f"==================================================")

    # 1. Create Space repo if it does not exist
    space_exists = False
    try:
        api.repo_info(repo_id=repo_id, repo_type="space")
        space_exists = True
        print(f"[+] Found existing Space '{repo_id}'.")
    except Exception:
        print(f"[*] Space '{repo_id}' does not exist yet. Attempting creation...")
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="space",
                space_sdk="gradio",
                private=private,
                exist_ok=True
            )
            space_exists = True
            print(f"[+] Hugging Face Space '{repo_id}' created successfully.")
        except Exception as e:
            print(f"\n[!] Notice: Automatic Space creation via API returned: {e}")
            print(f"\n👉 Hugging Face now requires Spaces to be created once through their Web UI:")
            print(f"   1. Go to: https://huggingface.co/new-space")
            print(f"   2. Space name: {space_name}")
            print(f"   3. Select SDK: 'Gradio'")
            print(f"   4. Choose free CPU tier and click 'Create Space'")
            print(f"   5. Once created, re-run this script to automatically upload all files!\n")
            return

    # 2. Stage files
    prepare_staging_directory(staging_dir)

    # 3. Upload folder to Space
    print(f"[*] Uploading files to https://huggingface.co/spaces/{repo_id}...")
    api.upload_folder(
        folder_path=staging_dir,
        repo_id=repo_id,
        repo_type="space",
        commit_message="Deploy TinyStories Mini-GPT Gradio app"
    )

    space_url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"\n🎉 Successfully deployed!")
    print(f"👉 Live Space URL: {space_url}")
    print(f"==================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy TinyStories Mini-GPT to Hugging Face Spaces")
    parser.add_argument("--space_name", type=str, default="tinystories-minigpt", help="Name of the Space on Hugging Face")
    parser.add_argument("--private", action="store_true", help="Set Space to private")
    args = parser.parse_args()

    deploy(space_name=args.space_name, private=args.private)
