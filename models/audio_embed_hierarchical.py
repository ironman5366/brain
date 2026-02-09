"""
Hierarchical EEG → Audio Embedding model for continuous/multi-second sequences.

Level 1 (Spatial Encoder): Per-second channel transformer (reuses baseline architecture).
Level 2 (Temporal Encoder): Transformer over the sequence of per-second embeddings.

Output: per-second audio embedding predictions, (B, W, 128, 75).
"""

from models.transformer import Transformer

import torch
from torch import nn
import torch.nn.functional as F
from pydantic import BaseModel
from huggingface_hub import PyTorchModelHubMixin
from accelerate import Accelerator
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class HierarchicalEEGAudioEmbedConfig(BaseModel):
    # Level 1: Spatial encoder (per 1-second chunk)
    encoder_dim: int = 512
    encoder_mlp_dim: int = 2048
    heads: int = 8
    depth: int = 6
    dim_head: int = 64
    dropout: float = 0.1
    num_channels: int = 120
    eeg_samples: int = 125

    # Level 2: Temporal encoder
    temporal_dim: int = 512
    temporal_depth: int = 4
    temporal_heads: int = 8
    temporal_dim_head: int = 64
    temporal_mlp_dim: int = 2048

    # Positional embeddings
    max_seq_len: int = 300  # max seconds (covers ~5 min songs)

    # Audio target (per second)
    audio_embed_dim: int = 128
    audio_frames: int = 75


class HierarchicalEEGAudioEmbed(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        *,
        encoder_dim: int,
        encoder_mlp_dim: int,
        heads: int,
        depth: int,
        dim_head: int,
        dropout: float,
        num_channels: int,
        eeg_samples: int,
        temporal_dim: int,
        temporal_depth: int,
        temporal_heads: int,
        temporal_dim_head: int,
        temporal_mlp_dim: int,
        max_seq_len: int,
        audio_embed_dim: int,
        audio_frames: int,
    ):
        super().__init__()
        self.audio_embed_dim = audio_embed_dim
        self.audio_frames = audio_frames
        self.encoder_dim = encoder_dim

        # --- Level 1: Spatial encoder (per 1-second chunk) ---
        self.channel_embed = nn.Linear(eeg_samples, encoder_dim)
        self.pos_embed = nn.Embedding(num_channels, encoder_dim)
        self.spatial_encoder = Transformer(
            dim=encoder_dim,
            depth=depth,
            heads=heads,
            mlp_dim=encoder_mlp_dim,
            dim_head=dim_head,
            dropout=dropout,
        )

        # --- Level 2: Temporal encoder (across seconds) ---
        self.temporal_project = nn.Linear(encoder_dim, temporal_dim)
        self.temporal_pos_embed = nn.Embedding(max_seq_len, temporal_dim)
        self.temporal_encoder = Transformer(
            dim=temporal_dim,
            depth=temporal_depth,
            heads=temporal_heads,
            mlp_dim=temporal_mlp_dim,
            dim_head=temporal_dim_head,
            dropout=dropout,
        )

        # --- Per-step output projection ---
        output_dim = audio_embed_dim * audio_frames  # 128 * 75 = 9600
        self.output_projection = nn.Sequential(
            nn.LayerNorm(temporal_dim),
            nn.Linear(temporal_dim, temporal_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_dim, output_dim),
        )

    def encode_spatial(self, eeg_chunk: torch.Tensor) -> torch.Tensor:
        """Encode a batch of 1-second chunks.

        Args:
            eeg_chunk: (B, C, T) where C=num_channels, T=eeg_samples

        Returns:
            (B, encoder_dim) pooled spatial embedding per chunk
        """
        tokens = self.channel_embed(eeg_chunk)  # (B, C, encoder_dim)
        positions = torch.arange(tokens.shape[1], device=eeg_chunk.device)
        tokens = tokens + self.pos_embed(positions)
        encoded = self.spatial_encoder(tokens)  # (B, C, encoder_dim)
        pooled = encoded.mean(dim=1)  # (B, encoder_dim)
        return pooled

    def forward(
        self,
        eeg: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            eeg: (B, W, C, T) multi-second EEG sequence
            lengths: (B,) optional actual sequence lengths for padding mask

        Returns:
            audio_pred: (B, W, audio_embed_dim, audio_frames) per-second predictions
        """
        B, W, C, T = eeg.shape

        # Level 1: Spatial encode each 1-second chunk
        eeg_flat = eeg.reshape(B * W, C, T)
        spatial_embeds = self.encode_spatial(eeg_flat)  # (B*W, encoder_dim)
        spatial_embeds = spatial_embeds.reshape(B, W, -1)  # (B, W, encoder_dim)

        # Level 2: Temporal encode
        temporal_input = self.temporal_project(spatial_embeds)  # (B, W, temporal_dim)
        positions = torch.arange(W, device=eeg.device)
        temporal_input = temporal_input + self.temporal_pos_embed(positions)

        # Build attention mask for variable-length sequences
        attn_mask = None
        if lengths is not None:
            # Create causal-style padding mask for SDPA
            # valid[i,j] = True if position j is within length for sample i
            valid = torch.arange(W, device=eeg.device).unsqueeze(0) < lengths.unsqueeze(1)  # (B, W)
            # SDPA expects (B, 1, W, W) or (B, H, W, W) mask
            # mask[b, :, i, j] = valid[b, j] (each query can attend to valid keys)
            attn_mask = valid.unsqueeze(1).unsqueeze(2).expand(B, 1, W, W).float()
            # Convert to additive mask: 0 for valid, -inf for invalid
            attn_mask = (1.0 - attn_mask) * torch.finfo(temporal_input.dtype).min

        temporal_encoded = self.temporal_encoder(
            temporal_input, attn_mask=attn_mask
        )  # (B, W, temporal_dim)

        # Per-step output projection
        audio_pred = self.output_projection(temporal_encoded)  # (B, W, 128*75)
        audio_pred = audio_pred.reshape(
            B, W, self.audio_embed_dim, self.audio_frames
        )  # (B, W, 128, 75)

        return audio_pred

    def load_spatial_from_baseline(self, baseline_checkpoint: str):
        """Initialize spatial encoder weights from a trained 1-second baseline."""
        from models.audio_embed import EEGAudioEmbed

        baseline = EEGAudioEmbed.from_pretrained(baseline_checkpoint)
        self.channel_embed.load_state_dict(baseline.channel_embed.state_dict())
        self.pos_embed.load_state_dict(baseline.pos_embed.state_dict())
        self.spatial_encoder.load_state_dict(baseline.encoder.state_dict())
        print(f"Loaded spatial encoder from {baseline_checkpoint}")

    @classmethod
    def from_config(cls, config: HierarchicalEEGAudioEmbedConfig):
        return cls(**config.model_dump())


class HierarchicalEEGAudioEmbedTrainer:
    def __init__(
        self,
        *,
        model: HierarchicalEEGAudioEmbed,
        accelerator: Accelerator,
        scheduler: LRScheduler,
        optimizer: Optimizer,
    ):
        self.model = model
        self.accelerator = accelerator
        self.scheduler = scheduler
        self.optimizer = optimizer

    def step(
        self,
        eeg: torch.Tensor,
        audio_embeds: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> dict:
        self.optimizer.zero_grad()

        audio_pred = self.model(eeg, lengths=lengths)  # (B, W, 128, 75)

        if lengths is not None:
            # Masked MSE: only valid positions
            B, W = audio_pred.shape[:2]
            mask = torch.arange(W, device=eeg.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(audio_pred)
            loss = F.mse_loss(audio_pred[mask], audio_embeds[mask])
        else:
            loss = F.mse_loss(audio_pred, audio_embeds)

        self.accelerator.backward(loss)
        self.optimizer.step()
        self.scheduler.step()

        self.accelerator.log({"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]})

        return {"loss": loss}
