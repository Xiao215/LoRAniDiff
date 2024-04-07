"""
This module implements the CLIP (Contrastive Language–Image Pre-training) model components used within
the latent diffusion model framework. It includes classes for token embeddings, the CLIP model layers,
and the CLIP model itself, structuring a simplified version of the CLIP architecture for embedding text
inputs into a latent space.
"""

import torch
from torch import nn
from torch.nn import functional as F
from ldm.module.attention import SelfAttention


class CLIPEmbedding(nn.Module):
    """Defines the embedding layer used in CLIP for encoding token and positional information."""

    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.position_embedding = nn.Parameter(torch.zeros((n_token, n_embd)))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embeds tokens and adds position embeddings."""
        token_embeddings = self.token_embedding(tokens)
        return token_embeddings + self.position_embedding


class CLIPLayer(nn.Module):
    """Implements a single layer of the CLIP model, consisting of a self-attention mechanism and feedforward neural network."""

    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies self-attention and feedforward layers with residual connections and layer normalization."""
        residue = x
        x = self.layernorm_1(x)
        x = self.attention(x)
        x += residue

        residue = x
        x = self.layernorm_2(x)
        x = F.gelu(self.linear_1(x))
        x = self.linear_2(x)
        x += residue

        return x


class CLIP(nn.Module):
    """The CLIP model, combining embeddings and multiple transformer layers for processing text input."""

    def __init__(
        self,
        n_vocab: int = 49408,
        n_embd: int = 768,
        n_layers: int = 12,
        n_token: int = 77,
    ):
        super().__init__()
        self.embedding = CLIPEmbedding(n_vocab, n_embd, n_token)
        self.layers = nn.ModuleList([CLIPLayer(12, n_embd) for _ in range(n_layers)])
        self.layernorm = nn.LayerNorm(n_embd)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        """Processes input tokens through the embedding layer and transformer layers."""
        tokens = tokens.type(torch.long)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        output = self.layernorm(state)
        return output
