"""
EEG → Audio Contrastive model.

CLIP-style contrastive learning: maps EEG and audio EnCodec embeddings
into a shared normalized embedding space using symmetric InfoNCE loss.
Optionally includes an MSE reconstruction head.
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


class EEGAudioContrastiveConfig(BaseModel):
    # EEG transformer encoder
    encoder_dim: int = 512
    encoder_mlp_dim: int = 2048
    heads: int = 8
    depth: int = 6
    dim_head: int = 64
    dropout: float = 0.1

    # EEG input
    num_channels: int = 120
    eeg_samples: int = 125

    # Audio input
    audio_embed_dim: int = 128
    audio_frames: int = 75

    # Audio encoder
    audio_encoder_depth: int = 4
    audio_encoder_heads: int = 8
    audio_encoder_dim_head: int = 64
    audio_encoder_mlp_dim: int = 2048

    # Contrastive learning
    shared_embed_dim: int = 256
    init_logit_scale: float = 2.6593  # ln(1/0.07)
    max_logit_scale: float = 4.6052  # ln(100)

    # Hybrid loss weights
    contrastive_weight: float = 1.0
    mse_weight: float = 0.5  # 0.0 = pure contrastive


class EEGAudioContrastive(nn.Module, PyTorchModelHubMixin):
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
        audio_encoder_depth: int,
        audio_encoder_heads: int,
        audio_encoder_dim_head: int,
        audio_encoder_mlp_dim: int,
        shared_embed_dim: int,
        init_logit_scale: float,
        max_logit_scale: float,
        contrastive_weight: float,
        mse_weight: float,
    ):
        super().__init__()
        self.audio_embed_dim = audio_embed_dim
        self.audio_frames = audio_frames
        self.max_logit_scale_val = max_logit_scale
        self.mse_weight = mse_weight

        # --- EEG branch ---
        self.channel_embed = nn.Linear(eeg_samples, encoder_dim)
        self.pos_embed = nn.Embedding(num_channels, encoder_dim)

        self.encoder = Transformer(
            dim=encoder_dim,
            depth=depth,
            heads=heads,
            mlp_dim=encoder_mlp_dim,
            dim_head=dim_head,
            dropout=dropout,
        )

        self.eeg_projector = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, encoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, shared_embed_dim),
        )

        # --- Audio branch ---
        self.frame_embed = nn.Linear(audio_embed_dim, encoder_dim)
        self.audio_pos_embed = nn.Embedding(audio_frames, encoder_dim)

        self.audio_encoder = Transformer(
            dim=encoder_dim,
            depth=audio_encoder_depth,
            heads=audio_encoder_heads,
            mlp_dim=audio_encoder_mlp_dim,
            dim_head=audio_encoder_dim_head,
            dropout=dropout,
        )

        self.audio_projector = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, encoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, shared_embed_dim),
        )

        # --- Learnable temperature ---
        self.log_logit_scale = nn.Parameter(torch.tensor(init_logit_scale))

        # --- Optional MSE reconstruction head ---
        if mse_weight > 0:
            output_dim = audio_embed_dim * audio_frames
            self.mse_head = nn.Sequential(
                nn.LayerNorm(encoder_dim),
                nn.Linear(encoder_dim, encoder_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(encoder_dim, output_dim),
            )
        else:
            self.mse_head = None

    def forward(
        self, eeg: torch.Tensor, audio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Args:
            eeg: (B, C, T) sparse EEG channels
            audio: (B, audio_embed_dim, audio_frames) EnCodec embeddings

        Returns:
            eeg_embed: (B, shared_embed_dim) L2-normalized EEG embeddings
            audio_embed: (B, shared_embed_dim) L2-normalized audio embeddings
            mse_pred: (B, audio_embed_dim, audio_frames) or None
        """
        # EEG branch
        tokens = self.channel_embed(eeg)  # (B, C, encoder_dim)
        positions = torch.arange(tokens.shape[1], device=eeg.device)
        tokens = tokens + self.pos_embed(positions)
        encoded = self.encoder(tokens)
        eeg_pooled = encoded.mean(dim=1)  # (B, encoder_dim)
        eeg_embed = F.normalize(self.eeg_projector(eeg_pooled), dim=-1)

        # Audio branch
        audio_t = audio.transpose(1, 2)  # (B, audio_frames, audio_embed_dim)
        audio_tokens = self.frame_embed(audio_t)  # (B, audio_frames, encoder_dim)
        audio_positions = torch.arange(audio_tokens.shape[1], device=audio.device)
        audio_tokens = audio_tokens + self.audio_pos_embed(audio_positions)
        audio_encoded = self.audio_encoder(audio_tokens)
        audio_pooled = audio_encoded.mean(dim=1)
        audio_embed = F.normalize(self.audio_projector(audio_pooled), dim=-1)

        # Optional MSE reconstruction
        mse_pred = None
        if self.mse_head is not None:
            mse_pred = self.mse_head(eeg_pooled)
            mse_pred = mse_pred.reshape(
                -1, self.audio_embed_dim, self.audio_frames
            )

        return eeg_embed, audio_embed, mse_pred

    @classmethod
    def from_config(cls, config: EEGAudioContrastiveConfig):
        return cls(**config.model_dump())


class EEGAudioContrastiveTrainer:
    def __init__(
        self,
        *,
        model: EEGAudioContrastive,
        accelerator: Accelerator,
        scheduler: LRScheduler,
        optimizer: Optimizer,
        contrastive_weight: float = 1.0,
        mse_weight: float = 0.5,
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.accelerator = accelerator
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.contrastive_weight = contrastive_weight
        self.mse_weight = mse_weight
        self.max_grad_norm = max_grad_norm

    def step(self, eeg: torch.Tensor, audio_embeds: torch.Tensor) -> dict:
        self.optimizer.zero_grad()

        eeg_embed, audio_embed, mse_pred = self.model(eeg, audio_embeds)

        # Contrastive loss (symmetric InfoNCE)
        logit_scale = self.model.log_logit_scale.exp().clamp(
            max=self.model.max_logit_scale_val
        )
        logits_eeg = logit_scale * eeg_embed @ audio_embed.T
        labels = torch.arange(eeg_embed.shape[0], device=eeg_embed.device)

        contrastive_loss = (
            F.cross_entropy(logits_eeg, labels)
            + F.cross_entropy(logits_eeg.T, labels)
        ) / 2

        total_loss = self.contrastive_weight * contrastive_loss

        # Optional MSE loss
        mse_loss_val = 0.0
        if mse_pred is not None and self.mse_weight > 0:
            mse_loss = F.mse_loss(mse_pred, audio_embeds)
            total_loss = total_loss + self.mse_weight * mse_loss
            mse_loss_val = mse_loss.item()

        self.accelerator.backward(total_loss)

        if self.max_grad_norm > 0:
            self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )

        self.optimizer.step()
        self.scheduler.step()

        # In-batch retrieval accuracy for monitoring
        with torch.no_grad():
            preds = logits_eeg.argmax(dim=-1)
            top1_acc = (preds == labels).float().mean().item()

        self.accelerator.log(
            {
                "loss": total_loss.item(),
                "contrastive_loss": contrastive_loss.item(),
                "mse_loss": mse_loss_val,
                "logit_scale": logit_scale.item(),
                "top1_acc": top1_acc,
                "lr": self.scheduler.get_last_lr()[0],
            }
        )

        return {"loss": total_loss, "top1_acc": top1_acc}
