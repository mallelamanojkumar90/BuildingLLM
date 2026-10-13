"""
model.py: Step 4 - Decoder-Only Transformer (Mini-GPT Architecture from Scratch).

Architecture Overview:
Input Tokens (B, T)
      │
      ▼
Token Embeddings + Positional Embeddings
      │
      ├───► [Transformer Block 1] ───┐
      │     - Pre-LayerNorm          │
      │     - Causal Self-Attention  │ Repeated N times
      │     - Residual Connection    │
      │     - Pre-LayerNorm          │
      │     - FeedForward MLP (GELU) │
      │     - Residual Connection ───┘
      │
      ├───► [Transformer Block N]
      │
      ▼
Final LayerNorm
      │
      ▼
Linear LM Head (Projects to vocab_size)
      │
      ▼
Logits (B, T, vocab_size)
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPTConfig:
    """
    Hyperparameters configuring the model architecture.
    """
    vocab_size: int = 4096       # Vocabulary size from tokenizer
    block_size: int = 128        # Maximum sequence length (context window T)
    n_layer: int = 4             # Number of Transformer blocks
    n_head: int = 4              # Number of attention heads
    n_embd: int = 128            # Embedding dimension (d_model)
    dropout: float = 0.1         # Dropout probability for regularization
    bias: bool = False           # Whether to include bias terms in linear layers


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention mechanism.
    
    Why Causal?
    In language modeling, the model predicts token t based ONLY on tokens 1 ... t-1.
    If token t were allowed to attend to token t+1, it would 'cheat' by looking ahead,
    causing training loss to collapse to zero while failing completely at generation.
    We enforce causality by applying a lower-triangular causal mask with -infinity
    for all future token positions prior to softmax.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # Linear projection for Query, Key, and Value in a single combined matrix
        # (B, T, n_embd) -> (B, T, 3 * n_embd)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        # Output projection back to n_embd
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask: lower-triangular matrix of 1s, shape (1, 1, block_size, block_size)
        # Registered as buffer so it moves with the model across devices and is not a trainable parameter
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size() # Batch, Sequence Length, Embedding Dim (n_embd)

        # 1. Project x into Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # 2. Reshape and transpose for multi-head attention:
        # (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 3. Scaled Dot-Product Attention:
        # Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V
        # att shape: (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # 4. Apply Causal Mask: Set future positions to -inf
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        
        # 5. Softmax over sequence length & dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # 6. Aggregate values: (B, n_head, T, T) @ (B, n_head, T, head_dim) -> (B, n_head, T, head_dim)
        y = att @ v

        # 7. Concatenate heads back to (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 8. Final linear projection and residual dropout
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    Consists of two linear layers with an expansion factor of 4 and GELU activation.
    GELU (Gaussian Error Linear Unit) provides smooth non-linearity, outperforming
    standard ReLU by permitting small negative gradients.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    A single Transformer Block with Pre-LayerNorm (Pre-LN) and Residual Connections.
    
    Pre-LN formulation:
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
        
    Why Pre-LN over Post-LN?
    Post-LN places LayerNorm after the residual addition. In deep networks, this dampens
    gradient signals reaching early layers, requiring delicate warmup. Pre-LN keeps
    the residual stream clean as an unobstructed highway, stabilizing gradient flow.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TinyStoriesGPT(nn.Module):
    """
    Complete Decoder-Only Generative Pre-trained Transformer.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        
        # Language model head projecting from hidden state to vocabulary logits
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight Tying: Tie token embedding weights and output head weights.
        # This reduces parameter footprint and improves generalization in small LLMs.
        self.lm_head.weight = self.transformer.wte.weight

        # Initialize weights with standard Gaussian distribution (mean=0, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Returns total number of model parameters.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Args:
            idx: Tensor of token indices, shape (B, T)
            targets: Optional ground-truth next token indices, shape (B, T)
        Returns:
            logits: Predicted next-token logits, shape (B, T, vocab_size)
            loss: Cross-Entropy loss if targets are provided, else None
        """
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds maximum context {self.config.block_size}"

        # 1. Positional indices: [0, 1, ..., T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        # 2. Add Token and Positional Embeddings
        tok_emb = self.transformer.wte(idx)           # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)           # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)  # (B, T, n_embd)

        # 3. Pass through N Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # 4. Final LayerNorm
        x = self.transformer.ln_f(x)                  # (B, T, n_embd)

        # 5. Output Projection to Vocabulary
        if targets is not None:
            # During training: compute logits and cross entropy loss
            logits = self.lm_head(x)                  # (B, T, vocab_size)
            # Flatten to calculate cross-entropy: (B*T, vocab_size) vs (B*T)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        else:
            # During inference: only compute logits for the last token to save computation!
            logits = self.lm_head(x[:, [-1], :])     # (B, 1, vocab_size)
            return logits, None

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        eos_id: int = None
    ) -> torch.Tensor:
        """
        Autoregressive generation loop.
        Takes conditioning tokens (B, T) and predicts max_new_tokens sequentially.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context if it exceeds max context window (block_size)
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass: get logits for the last token
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # (B, vocab_size)

            # Apply Temperature scaling (lower = more deterministic, higher = more random)
            logits = logits / max(temperature, 1e-5)

            # Optional Top-K Filtering
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Optional Top-P (Nucleus) Filtering
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Mask out tokens with cumulative probability above top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift mask right so at least one token is kept
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float("Inf")

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample next token from probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

            # Stop if EOS token is generated
            if eos_id is not None and idx_next.item() == eos_id:
                break

        return idx

if __name__ == "__main__":
    # Test model instantiation and parameter count
    cfg = GPTConfig(
        vocab_size=4096,
        block_size=128,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.1
    )
    model = TinyStoriesGPT(cfg)
    print(f"[+] Instantiated TinyStoriesGPT!")
    print(f"    Total Parameters: {model.get_num_params():,}")
    print(f"    Non-Embedding Parameters: {model.get_num_params(non_embedding=True):,}")
    
    # Test dummy forward pass
    dummy_input = torch.randint(0, cfg.vocab_size, (2, 32))
    dummy_targets = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, loss = model(dummy_input, dummy_targets)
    print(f"    Input shape: {dummy_input.shape}")
    print(f"    Logits shape: {logits.shape}")
    print(f"    Initial Loss: {loss.item():.4f} (Expected random ~ ln({cfg.vocab_size}) = {math.log(cfg.vocab_size):.4f})")
