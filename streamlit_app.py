"""
streamlit_app.py: Streamlit Web Application for TinyStories Mini-GPT.
Can be deployed 100% FREE on Streamlit Community Cloud (https://share.streamlit.io).
"""

import os
import time
import torch
import torch.nn.functional as F
import streamlit as st
from huggingface_hub import hf_hub_download

from model import TinyStoriesGPT, GPTConfig
from tokenizer_util import StoryTokenizer

st.set_page_config(
    page_title="TinyStories Mini-GPT",
    page_icon="📖",
    layout="centered"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ID = "mallelamanoj75/tinystories-minigpt"
LOCAL_CKPT = os.path.join("checkpoints", "best_model.pt")
TOKENIZER_DIR = "tokenizer_data"

@st.cache_resource
def load_resources():
    """
    Loads tokenizer and model weights.
    If best_model.pt is not found locally, automatically downloads it from Hugging Face Models!
    """
    # 1. Setup tokenizer files
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    vocab_path = os.path.join(TOKENIZER_DIR, "vocab.json")
    merges_path = os.path.join(TOKENIZER_DIR, "merges.txt")
    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        hf_hub_download(repo_id=REPO_ID, filename="vocab.json", local_dir=TOKENIZER_DIR)
        hf_hub_download(repo_id=REPO_ID, filename="merges.txt", local_dir=TOKENIZER_DIR)

    tok = StoryTokenizer(TOKENIZER_DIR)

    # 2. Setup model checkpoint
    if os.path.exists(LOCAL_CKPT):
        ckpt_file = LOCAL_CKPT
    else:
        st.info("Downloading model weights from Hugging Face Model Hub...")
        ckpt_file = hf_hub_download(repo_id=REPO_ID, filename="best_model.pt")

    checkpoint = torch.load(ckpt_file, map_location=DEVICE, weights_only=False)
    config = checkpoint["config"]
    m = TinyStoriesGPT(config).to(DEVICE)
    m.load_state_dict(checkpoint["state_dict"])
    m.eval()
    return m, tok

def stream_story(
    m: TinyStoriesGPT,
    tok: StoryTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float
):
    prompt_ids = tok.encode(prompt.strip())
    if len(prompt_ids) == 0:
        prompt_ids = [tok.tokenizer.token_to_id("Once")]

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)

    for _ in range(int(max_new_tokens)):
        idx_cond = idx if idx.size(1) <= m.config.block_size else idx[:, -m.config.block_size:]

        logits, _ = m(idx_cond)
        logits = logits[:, -1, :].clone()

        # Repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(idx[0].tolist()):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

        # Temperature
        logits = logits / max(temperature, 1e-5)

        # Top-K
        if top_k > 0:
            v, _ = torch.topk(logits, min(int(top_k), logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        # Top-P
        if 0.0 < top_p < 1.0:
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

        token_str = tok.decode([idx_next.item()])
        if "<|endoftext|>" in token_str:
            yield "\n\n**[End of Story]**"
            break

        yield token_str

# --- UI Layout ---
st.title("📖 TinyStories Mini-GPT")
st.caption("A lightweight 1.3M-parameter Transformer trained from scratch on Microsoft Research's TinyStories dataset.")

# Sidebar Settings
st.sidebar.header("⚙️ Generation Settings")
temperature = st.sidebar.slider("Temperature (Creativity)", 0.1, 1.5, 0.8, 0.05)
max_new_tokens = st.sidebar.slider("Max New Tokens", 20, 250, 120, 10)
top_k = st.sidebar.slider("Top-K", 1, 100, 30, 1)
top_p = st.sidebar.slider("Top-P (Nucleus)", 0.1, 1.0, 0.9, 0.05)
repetition_penalty = st.sidebar.slider("Repetition Penalty", 1.0, 1.5, 1.15, 0.05)

# Example selector
example_prompts = [
    "Custom Prompt",
    "Once upon a time, Lily found a sparkling green stone in the garden",
    "Once upon a time, Timmy found a little puppy in the backyard",
    "One sunny morning, Mia and her mom went to the big park",
    "There was once a little bird that wanted to fly over the clouds"
]
selected_example = st.selectbox("Quick Examples", example_prompts)

default_text = "Once upon a time, Lily found a sparkling green stone in the garden" if selected_example == "Custom Prompt" else selected_example
prompt = st.text_area("Story Prompt", value=default_text, height=100)

if st.button("✨ Generate Story", type="primary"):
    with st.spinner("Loading model and generating..."):
        m, tok = load_resources()

    st.subheader("Generated Story:")
    story_container = st.empty()
    full_text = prompt.strip() + " "

    generator = stream_story(
        m=m,
        tok=tok,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty
    )

    for chunk in generator:
        full_text += chunk
        story_container.markdown(full_text)
        time.sleep(0.01)

st.markdown("---")
st.markdown("""
### 🔍 Model Architecture
- **Parameters:** ~1.33 Million (4 layers, 4 attention heads, $d_{model} = 128$)
- **Tokenizer:** Custom Byte-Level BPE (4,096 tokens)
- **Model Registry:** [Hugging Face Models (mallelamanoj75/tinystories-minigpt)](https://huggingface.co/mallelamanoj75/tinystories-minigpt)
- **Source Code:** [GitHub (mallelamanojkumar90/BuildingLLM)](https://github.com/mallelamanojkumar90/BuildingLLM)
""")
