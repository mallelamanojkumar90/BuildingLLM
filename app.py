"""
app.py: Gradio Web Application for TinyStories Mini-GPT.
Can be run locally or deployed directly to Hugging Face Spaces.
"""

import os
import torch
import torch.nn.functional as F
import gradio as gr

from model import TinyStoriesGPT, GPTConfig
from tokenizer_util import StoryTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = os.path.join("checkpoints", "best_model.pt")
TOKENIZER_DIR = "tokenizer_data"

CUSTOM_CSS = """
.gradio-container { max-width: 850px; margin: auto; }
#story-output textarea { font-family: 'Georgia', serif; font-size: 1.1rem; line-height: 1.6; }
"""

# Global model and tokenizer handles
model = None
tokenizer = None

def load_model_and_tokenizer():
    global model, tokenizer
    if model is not None and tokenizer is not None:
        return model, tokenizer

    print(f"[*] Loading tokenizer from {TOKENIZER_DIR}...")
    tokenizer = StoryTokenizer(TOKENIZER_DIR)

    print(f"[*] Loading model checkpoint from {CHECKPOINT_PATH} (Device: {DEVICE})...")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found at '{CHECKPOINT_PATH}'. Please train the model first using train.py!"
        )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    config = checkpoint["config"]
    model = TinyStoriesGPT(config).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"[+] Model loaded successfully ({model.get_num_params():,} parameters).")
    return model, tokenizer

@torch.no_grad()
def generate_story_streaming(
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.8,
    top_k: int = 30,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15
):
    """
    Generates story text with streaming yields for real-time typing effect in Gradio.
    """
    if not prompt or prompt.strip() == "":
        prompt = "Once upon a time"

    m, tok = load_model_and_tokenizer()

    # Encode prompt
    prompt_ids = tok.encode(prompt.strip())
    if len(prompt_ids) == 0:
        prompt_ids = [tok.tokenizer.token_to_id("Once")]

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    current_text = prompt.strip()

    for _ in range(int(max_new_tokens)):
        # Crop context if it exceeds block_size (128)
        idx_cond = idx if idx.size(1) <= m.config.block_size else idx[:, -m.config.block_size:]

        # Forward pass: get logits for last position
        logits, _ = m(idx_cond)
        logits = logits[:, -1, :].clone() # (1, vocab_size)

        # Repetition Penalty: discount logits of already seen tokens to prevent loops
        if repetition_penalty != 1.0:
            for token_id in set(idx[0].tolist()):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

        # Temperature scaling
        logits = logits / max(temperature, 1e-5)

        # Top-K filtering
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(int(top_k), logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        # Top-P (Nucleus) filtering
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float("Inf")

        # Probability distribution & sampling
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # Append next token
        idx = torch.cat((idx, idx_next), dim=1)

        # Decode accumulated tokens
        decoded = tok.decode(idx[0].tolist())

        # If model generated end-of-text, cleanly terminate without bleeding into next story
        if "<|endoftext|>" in decoded:
            clean_story = decoded.split("<|endoftext|>")[0].strip()
            current_text = clean_story + "\n\n[End of Story]"
            yield current_text
            break

        current_text = decoded
        yield current_text


# Build Gradio Interface
def create_app():
    title = "📖 TinyStories Mini-GPT"
    description = """
    A lightweight **Decoder-Only Transformer (1.3M parameters)** built and trained from scratch in pure PyTorch 
    on Microsoft Research's **TinyStories** dataset. 
    
    Type a story starter or pick an example below to watch the model generate a children's story in real time!
    """

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(description)

        with gr.Row():
            with gr.Column(scale=5):
                prompt_input = gr.Textbox(
                    label="Story Prompt",
                    placeholder="Once upon a time, there was a little boy named Timmy...",
                    lines=3,
                    value="Once upon a time, Lily found a sparkling green stone in the garden"
                )

                with gr.Accordion("⚙️ Generation Settings", open=False):
                    max_tokens_slider = gr.Slider(
                        minimum=10, maximum=300, value=120, step=5, label="Max New Tokens"
                    )
                    temp_slider = gr.Slider(
                        minimum=0.1, maximum=1.5, value=0.8, step=0.05,
                        label="Temperature (Creativity)",
                        info="Lower is more conservative; higher is more adventurous."
                    )
                    top_k_slider = gr.Slider(
                        minimum=1, maximum=100, value=30, step=1,
                        label="Top-K",
                        info="Restricts sampling to the top K most likely words."
                    )
                    top_p_slider = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.9, step=0.05,
                        label="Top-P (Nucleus)",
                        info="Keeps smallest pool of tokens covering P% cumulative probability."
                    )
                    rep_penalty_slider = gr.Slider(
                        minimum=1.0, maximum=1.5, value=1.15, step=0.05,
                        label="Repetition Penalty",
                        info="Penalizes repeating recently generated words (1.0 = none, 1.15 = balanced)."
                    )

                with gr.Row():
                    generate_btn = gr.Button("✨ Generate Story", variant="primary", scale=2)
                    clear_btn = gr.Button("Clear", scale=1)

            with gr.Column(scale=5):
                story_output = gr.Textbox(
                    label="Generated Story",
                    elem_id="story-output",
                    lines=10,
                    interactive=False
                )

        gr.Examples(
            examples=[
                ["Once upon a time, Lily found a sparkling green stone in the garden"],
                ["Once upon a time, Timmy found a little puppy in the backyard"],
                ["One sunny morning, Mia and her mom went to the big park"],
                ["There was once a little bird that wanted to fly over the clouds"]
            ],
            inputs=[prompt_input],
            label="Click an Example to Load:"
        )

        generate_btn.click(
            fn=generate_story_streaming,
            inputs=[prompt_input, max_tokens_slider, temp_slider, top_k_slider, top_p_slider, rep_penalty_slider],
            outputs=story_output
        )

        clear_btn.click(
            fn=lambda: ("", ""),
            inputs=None,
            outputs=[prompt_input, story_output]
        )

        gr.Markdown("""
        ---
        ### 🔍 Architecture Summary
        - **Parameters:** ~1.33 Million (4 layers, 4 attention heads, $d_{model} = 128$)
        - **Tokenizer:** Custom Byte-Level BPE (4,096 tokens)
        - **Context Window:** 128 tokens
        - **Corpus:** Microsoft Research TinyStories (synthesized with 3- to 4-year-old vocabulary)
        """)

    return demo

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run TinyStories Mini-GPT Web App")
    parser.add_argument("--share", action="store_true", help="Create a free public gradio.live link")
    parser.add_argument("--port", type=int, default=7860, help="Local server port")
    args = parser.parse_args()

    demo = create_app()
    demo.launch(
        server_name="0.0.0.0" if args.share else "127.0.0.1",
        server_port=args.port,
        css=CUSTOM_CSS,
        share=args.share
    )
