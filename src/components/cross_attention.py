import torch
import torch.nn as nn

class ContextCrossAttn(nn.Module):
    """
    Cross-attention from word embeddings (query) to context embeddings (key/value),
    followed by a learned gate controlling how much context shifts each word.

    word_embeds:    (B, N, D)  -- pure BiLM word representations
    context_embeds: (B, L, D)  -- encoded retrieved context tokens
    context_mask:   (B, L)     -- True = ignore (padding)  [optional]

    Returns enriched word_embeds of same shape (B, N, D).
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        word_embeds: torch.Tensor,
        context_embeds: torch.Tensor,
        context_mask: torch.Tensor = None,
    ) -> torch.Tensor:

        attended, _ = self.cross_attn(
            query=word_embeds,
            key=context_embeds,
            value=context_embeds,
            key_padding_mask=context_mask,
        )

        g = self.gate(word_embeds)
        enriched = g * attended + (1 - g) * word_embeds

        return self.norm(enriched)