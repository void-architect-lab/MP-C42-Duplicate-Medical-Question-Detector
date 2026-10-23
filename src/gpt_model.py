"""Course-style GPT-2 Small implementation for binary question-pair classification.

The transformer architecture is implemented with PyTorch modules (no transformers
pipeline/Trainer/AutoModel). Pretrained GPT-2 weights are downloaded separately and
mapped into this model's own layers.
"""
from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn


GPT2_SMALL_CONFIG: Dict[str, int | float | bool] = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.0,
    "qkv_bias": True,
}


class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps) * self.scale + self.shift


class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The tanh approximation matches the GPT-2 GELU variant.
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, context_length: int, num_heads: int,
                 dropout: float, qkv_bias: bool) -> None:
        super().__init__()
        if d_out % num_heads != 0:
            raise ValueError("Embedding dimension must be divisible by the number of heads")
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = x.shape
        if num_tokens > self.mask.shape[0]:
            raise ValueError(f"Sequence length {num_tokens} exceeds context length {self.mask.shape[0]}")
        keys = self.W_key(x).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.W_query(x).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.W_value(x).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        scores = queries @ keys.transpose(2, 3)
        scores = scores.masked_fill(self.mask[:num_tokens, :num_tokens], -torch.inf)
        weights = torch.softmax(scores / math.sqrt(self.head_dim), dim=-1)
        weights = self.dropout(weights)
        context = (weights @ values).transpose(1, 2).contiguous()
        context = context.view(batch_size, num_tokens, -1)
        return self.out_proj(context)


class FeedForward(nn.Module):
    def __init__(self, cfg: Dict[str, int | float | bool]) -> None:
        super().__init__()
        emb_dim = int(cfg["emb_dim"])
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Dict[str, int | float | bool]) -> None:
        super().__init__()
        emb_dim = int(cfg["emb_dim"])
        self.att = MultiHeadAttention(
            d_in=emb_dim,
            d_out=emb_dim,
            context_length=int(cfg["context_length"]),
            num_heads=int(cfg["n_heads"]),
            dropout=float(cfg["drop_rate"]),
            qkv_bias=bool(cfg["qkv_bias"]),
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(emb_dim)
        self.norm2 = LayerNorm(emb_dim)
        self.drop_shortcut = nn.Dropout(float(cfg["drop_rate"]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        return x + shortcut


class GPTModel(nn.Module):
    """GPT-2-like backbone with a replaceable binary classification output head."""
    def __init__(self, cfg: Dict[str, int | float | bool], num_classes: int = 2) -> None:
        super().__init__()
        self.cfg = dict(cfg)
        emb_dim = int(cfg["emb_dim"])
        self.tok_emb = nn.Embedding(int(cfg["vocab_size"]), emb_dim)
        self.pos_emb = nn.Embedding(int(cfg["context_length"]), emb_dim)
        self.drop_emb = nn.Dropout(float(cfg["drop_rate"]))
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(int(cfg["n_layers"]))])
        self.final_norm = LayerNorm(emb_dim)
        self.out_head = nn.Linear(emb_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if seq_len > int(self.cfg["context_length"]):
            raise ValueError("Input sequence exceeds configured context length")
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)  # logits at every token; caller selects each pair's last real token


def load_gpt2_small_weights(model: GPTModel) -> str:
    """Download GPT-2 Small parameters and map them into this own GPTModel.

    Uses Hugging Face Hub only as a file host for the official GPT-2 state dict;
    it does not use AutoModel, Trainer, or pipeline for the core model.
    """
    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt first (huggingface_hub and safetensors are required).") from exc

    path = hf_hub_download(repo_id="openai-community/gpt2", filename="model.safetensors")
    state = load_file(path, device="cpu")

    # NORMALIZE_HF_GPT2_KEY_PREFIXES: accept common official checkpoint prefixes.
    normalized_state = {}
    for original_key, tensor in state.items():
        key = original_key
        if key.startswith("model.transformer."):
            key = key[len("model."):]
        elif key.startswith("model."):
            key = key[len("model."):]
        if key.startswith(("wte.", "wpe.", "h.", "ln_f.")):
            key = "transformer." + key
        normalized_state[key] = tensor
    state = normalized_state

    # GPT-2 ties the token embedding to lm_head. Some recent Hugging Face
    # safetensors exports keep only lm_head.weight for this shared tensor.
    # Restore the alias expected by the course-style loader.
    if "transformer.wte.weight" not in state:
        tied_candidates = (
            "lm_head.weight",
            "model.transformer.wte.weight",
            "wte.weight",
            "model.wte.weight",
        )
        tied_key = next((key for key in tied_candidates if key in state), None)
        if tied_key is None:
            first_keys = ", ".join(list(state.keys())[:20])
            raise KeyError(
                "Could not locate GPT-2 token embeddings in model.safetensors. "
                f"Available first keys: {first_keys}"
            )
        state["transformer.wte.weight"] = state[tied_key]


    def copy_(target: torch.Tensor, source: torch.Tensor, transpose: bool = False) -> None:
        value = source.T.contiguous() if transpose else source.contiguous()
        if tuple(target.shape) != tuple(value.shape):
            raise ValueError(f"GPT-2 weight shape mismatch: expected {tuple(target.shape)}, got {tuple(value.shape)}")
        target.copy_(value)

    with torch.no_grad():
        copy_(model.tok_emb.weight, state["transformer.wte.weight"])
        copy_(model.pos_emb.weight, state["transformer.wpe.weight"])
        for i, block in enumerate(model.trf_blocks):
            base = f"transformer.h.{i}."
            for src_suffix, target_norm in (("ln_1", block.norm1), ("ln_2", block.norm2)):
                copy_(target_norm.scale, state[f"{base}{src_suffix}.weight"])
                copy_(target_norm.shift, state[f"{base}{src_suffix}.bias"])

            qkv_weight = state[f"{base}attn.c_attn.weight"].T.contiguous().chunk(3, dim=0)
            qkv_bias = state[f"{base}attn.c_attn.bias"].chunk(3, dim=0)
            for layer, weight, bias in zip(
                (block.att.W_query, block.att.W_key, block.att.W_value), qkv_weight, qkv_bias
            ):
                copy_(layer.weight, weight)
                copy_(layer.bias, bias)

            copy_(block.att.out_proj.weight, state[f"{base}attn.c_proj.weight"], transpose=True)
            copy_(block.att.out_proj.bias, state[f"{base}attn.c_proj.bias"])
            copy_(block.ff.layers[0].weight, state[f"{base}mlp.c_fc.weight"], transpose=True)
            copy_(block.ff.layers[0].bias, state[f"{base}mlp.c_fc.bias"])
            copy_(block.ff.layers[2].weight, state[f"{base}mlp.c_proj.weight"], transpose=True)
            copy_(block.ff.layers[2].bias, state[f"{base}mlp.c_proj.bias"])

        copy_(model.final_norm.scale, state["transformer.ln_f.weight"])
        copy_(model.final_norm.shift, state["transformer.ln_f.bias"])
    del state
    return path


def freeze_for_classification(model: GPTModel) -> int:
    """Freeze the pretrained backbone, then unfreeze the last block and classifier head."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.trf_blocks[-1].parameters():
        parameter.requires_grad = True
    for parameter in model.final_norm.parameters():
        parameter.requires_grad = True
    for parameter in model.out_head.parameters():
        parameter.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
