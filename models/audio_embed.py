"""
EEG → Audio Embedding model.

Maps EEG signals to EnCodec encoder embeddings, learning to predict
continuous audio representations from brain activity.
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


class EEGAudioEmbedConfig(BaseModel):
    # Transformer encoder
    encoder_dim: int = 512
    encoder_mlp_dim: int = 2048
    heads: int = 8
    depth: int = 6
    dim_head: int = 64
    dropout: float = 0.1

    # EEG input
    num_channels: int = 120
    eeg_samples: int = 125

    # Audio embedding target
    audio_embed_dim: int = 128
    audio_frames: int = 75


class EEGAudioEmbed(nn.Module, PyTorchModelHubMixin):
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
        audio_embed_dim: int,
        audio_frames: int,
    ):
        super().__init__()
        self.audio_embed_dim = audio_embed_dim
        self.audio_frames = audio_frames

        # Tokenize: project each channel's time samples to encoder_dim
        # Input (B, C, T) -> tokens (B, C, encoder_dim)
        self.channel_embed = nn.Linear(eeg_samples, encoder_dim)
        self.pos_embed = nn.Embedding(num_channels, encoder_dim)

        # Transformer encoder
        self.encoder = Transformer(
            dim=encoder_dim,
            depth=depth,
            heads=heads,
            mlp_dim=encoder_mlp_dim,
            dim_head=dim_head,
            dropout=dropout,
        )

        # Projection head: encoded EEG -> audio embedding space
        output_dim = audio_embed_dim * audio_frames
        self.projection = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, encoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, output_dim),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg: (B, C, T) sparse EEG channels

        Returns:
            audio_pred: (B, audio_embed_dim, audio_frames) predicted EnCodec embeddings
        """
        # Channel tokenization
        tokens = self.channel_embed(eeg)  # (B, C, encoder_dim)

        # Add positional embeddings
        positions = torch.arange(tokens.shape[1], device=eeg.device)
        tokens = tokens + self.pos_embed(positions)

        # Transformer encode
        encoded = self.encoder(tokens)  # (B, C, encoder_dim)

        # Mean pool over channel tokens
        pooled = encoded.mean(dim=1)  # (B, encoder_dim)

        # Project to audio embedding space
        audio_pred = self.projection(pooled)  # (B, audio_embed_dim * audio_frames)
        audio_pred = audio_pred.reshape(
            -1, self.audio_embed_dim, self.audio_frames
        )  # (B, 128, 75)

        return audio_pred

    @classmethod
    def from_config(cls, config: EEGAudioEmbedConfig):
        return cls(**config.model_dump())


class EEGAudioEmbedTrainer:
    def __init__(
        self,
        *,
        model: EEGAudioEmbed,
        accelerator: Accelerator,
        scheduler: LRScheduler,
        optimizer: Optimizer,
    ):
        self.model = model
        self.accelerator = accelerator
        self.scheduler = scheduler
        self.optimizer = optimizer

    def step(self, eeg: torch.Tensor, audio_embeds: torch.Tensor) -> dict:
        self.optimizer.zero_grad()

        audio_pred = self.model(eeg)
        loss = F.mse_loss(audio_pred, audio_embeds)

        self.accelerator.backward(loss)
        self.optimizer.step()
        self.scheduler.step()

        self.accelerator.log({"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]})

        return {"loss": loss}

    @torch.no_grad()
    def eval_batch(self, eeg: torch.Tensor, audio_embeds: torch.Tensor) -> dict:
        audio_pred = self.model(eeg)
        loss = F.mse_loss(audio_pred, audio_embeds)

        pred_flat = audio_pred.reshape(audio_pred.shape[0], -1)
        target_flat = audio_embeds.reshape(audio_embeds.shape[0], -1)
        cosine_sim = F.cosine_similarity(pred_flat, target_flat, dim=1).mean().item()

        return {
            "loss": loss.item(),
            "cosine_sim": cosine_sim,
            "batch_size": eeg.shape[0],
        }
