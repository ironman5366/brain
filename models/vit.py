# Modified from luicdrains' implementation (https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py) for EEG data

# Builtin imports
from typing import Literal

# Internal imports
from constants import NUM_CHANNELS
from models.transformer import Transformer

# External imports
import torch
from torch import nn
from torch.nn import Module
from einops.layers.torch import Rearrange
from einops import repeat
from pydantic import BaseModel

DEFAULT_POOL = "cls"
DEFAULT_DROUPOUT = 0.0
DEFAULT_DIM_HEAD = 64
DEFAULT_EMB_DROPOUT = 0.0


class EEGViTConfig(BaseModel):
    sample_len: int
    patch_len: int
    num_classes: int
    dim: int
    depth: int
    heads: int
    mlp_dim: int

    pool: Literal["cls"] | Literal["mean"] = DEFAULT_POOL
    channels: int = NUM_CHANNELS
    dropout: float = DEFAULT_DROUPOUT
    dim_head: int = DEFAULT_DIM_HEAD
    emb_dropout: float = DEFAULT_EMB_DROPOUT


class EEGViT(Module):
    def __init__(
        self,
        *,
        sample_len,
        patch_len,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool=DEFAULT_POOL,
        channels=NUM_CHANNELS,
        dim_head=DEFAULT_DIM_HEAD,
        dropout=DEFAULT_DROUPOUT,
        emb_dropout=DEFAULT_EMB_DROPOUT,
    ):
        super().__init__()

        assert sample_len % patch_len == 0, (
            f"Sample length ({sample_len}) must be divisible by the patch length ({patch_len})"
        )

        # Patch *temporally*, so given a sample len of 256, and a patch_len of 16, we'll generate 16 patch tokens
        num_patches = sample_len // patch_len
        patch_dim = channels * patch_len

        assert pool in {"cls", "mean"}, (
            "pool type must be either cls (cls token) or mean (mean pooling)"
        )
        num_cls_tokens = 1 if pool == "cls" else 0

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b ch (np pl) -> b np (pl ch)",
                pl=patch_len,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.cls_token = nn.Parameter(torch.randn(num_cls_tokens, dim))

        # Temporal embedding
        self.temporal_embedding = nn.Parameter(
            torch.randn(num_patches + num_cls_tokens, dim)
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool

        # TODO: kept in just so code would scan similarily to the lucidrains OG, but why have this?
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, sample):
        batch = sample.shape[0]
        x = self.to_patch_embedding(sample)

        cls_tokens = repeat(self.cls_token, "... d -> b ... d", b=batch)
        x = torch.cat((cls_tokens, x), dim=1)

        seq = x.shape[1]

        x = x + self.temporal_embedding[:seq]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

    @classmethod
    def from_config(cls, config: EEGViTConfig):
        return cls(**config.model_dump())
