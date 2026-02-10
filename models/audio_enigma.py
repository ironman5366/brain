"""
EEG → Audio ENIGMA-style model.

Replicates the ENIGMA architecture (Spatio-Temporal CNN + MLP Projector)
but targets flattened EnCodec audio embeddings instead of CLIP image embeddings.

Reference: /kreka/research/willy/side/ENIGMA/source/models.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from pydantic import BaseModel
from huggingface_hub import PyTorchModelHubMixin
from accelerate import Accelerator
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class EEGAudioENIGMAConfig(BaseModel):
    # EEG input
    num_channels: int = 32
    eeg_samples: int = 125

    # Audio target
    audio_embed_dim: int = 128
    audio_frames: int = 75
    embed_dim: int = 9600  # 128 * 75 flattened EnCodec

    # MLP projector
    proj_dropout: float = 0.5

    # Contrastive learning
    init_logit_scale: float = 0.0
    max_logit_scale: float = 4.6052  # ln(100)

    # Loss weights (trainer-only)
    mse_weight: float = 1.0
    contrastive_weight: float = 0.5
    max_grad_norm: float = 1.0

    # Optional subject-wise linear (None = disabled)
    subject_ids: list[str] | None = None


# --- Sub-modules (from ENIGMA) ---


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)


class SpatioTemporalCNN(nn.Module):
    """Spatio-temporal CNN encoder from ENIGMA."""

    def __init__(
        self,
        emb_size: int = 4,
        conv1_kernel: tuple[int, int] = (1, 5),
        pool1_kernel: tuple[int, int] = (1, 17),
        pool1_stride: tuple[int, int] = (1, 5),
        conv2_kernel: tuple[int, int] = (32, 1),
    ):
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, conv1_kernel, stride=(1, 1)),
            nn.AvgPool2d(pool1_kernel, pool1_stride),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, conv2_kernel, stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (B, C, T) -> (B, 1, C, T)
        x = self.tsconv(x)
        x = self.projection(x)
        x = x.contiguous().view(x.size(0), -1)  # (B, hidden_dim)
        return x


class MLPProjector(nn.Sequential):
    """MLP projection head from ENIGMA with residual connection."""

    def __init__(self, hidden_dim: int, embed_dim: int, dropout: float = 0.5):
        super().__init__(
            nn.Linear(hidden_dim, embed_dim),
            ResidualAdd(
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(embed_dim, embed_dim),
                    nn.Dropout(dropout),
                )
            ),
            nn.LayerNorm(embed_dim),
        )


# --- Main model ---


class EEGAudioENIGMA(nn.Module, PyTorchModelHubMixin):
    _TRAINER_ONLY_FIELDS = {"mse_weight", "contrastive_weight", "max_grad_norm"}

    def __init__(
        self,
        *,
        num_channels: int,
        eeg_samples: int,
        audio_embed_dim: int,
        audio_frames: int,
        embed_dim: int,
        proj_dropout: float,
        init_logit_scale: float,
        max_logit_scale: float,
        subject_ids: list[str] | None = None,
    ):
        super().__init__()
        self.audio_embed_dim = audio_embed_dim
        self.audio_frames = audio_frames
        self.max_logit_scale_val = max_logit_scale

        # Optional subject-wise linear
        if subject_ids is not None:
            self.subject_wise_linear = nn.ModuleDict(
                {sid: nn.Linear(eeg_samples, eeg_samples) for sid in subject_ids}
            )
        else:
            self.subject_wise_linear = None

        # Spatio-temporal CNN encoder
        self.tsencoder = SpatioTemporalCNN(
            emb_size=4,
            conv1_kernel=(1, 5),
            pool1_kernel=(1, 17),
            pool1_stride=(1, 5),
            conv2_kernel=(num_channels, 1),
        )

        # Compute hidden_dim dynamically from CNN output
        with torch.no_grad():
            dummy = torch.zeros(1, num_channels, eeg_samples)
            hidden_dim = self.tsencoder(dummy).shape[1]

        # MLP projector to audio embedding space
        self.mlp_proj = MLPProjector(hidden_dim, embed_dim, dropout=proj_dropout)

        # Learnable temperature for contrastive loss
        self.log_logit_scale = nn.Parameter(torch.tensor(init_logit_scale))

    def forward(self, eeg: torch.Tensor, subjects=None) -> torch.Tensor:
        """
        Args:
            eeg: (B, C, T) EEG input
            subjects: optional list of subject IDs (len B)

        Returns:
            (B, embed_dim) predicted audio embedding
        """
        # Optional subject-wise preprocessing
        if self.subject_wise_linear is not None and subjects is not None:
            x = torch.zeros_like(eeg)
            subjects_arr = np.array(subjects)
            for sid in np.unique(subjects_arr):
                mask = torch.from_numpy(subjects_arr == sid)
                x[mask] = self.subject_wise_linear[sid](eeg[mask])
        else:
            x = eeg

        z = self.tsencoder(x)       # (B, hidden_dim)
        c = self.mlp_proj(z)        # (B, embed_dim)
        return c

    @classmethod
    def from_config(cls, config: EEGAudioENIGMAConfig):
        d = {
            k: v
            for k, v in config.model_dump().items()
            if k not in cls._TRAINER_ONLY_FIELDS
        }
        return cls(**d)


# --- Trainer ---


class EEGAudioENIGMATrainer:
    def __init__(
        self,
        *,
        model: EEGAudioENIGMA,
        accelerator: Accelerator,
        scheduler: LRScheduler,
        optimizer: Optimizer,
        mse_weight: float = 1.0,
        contrastive_weight: float = 0.5,
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.accelerator = accelerator
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.mse_weight = mse_weight
        self.contrastive_weight = contrastive_weight
        self.max_grad_norm = max_grad_norm

    def step(self, eeg: torch.Tensor, audio_embeds: torch.Tensor) -> dict:
        self.optimizer.zero_grad()

        # Flatten audio: (B, 128, 75) -> (B, 9600)
        audio_flat = audio_embeds.reshape(audio_embeds.shape[0], -1)

        # Forward
        output = self.model(eeg)  # (B, embed_dim)

        # MSE loss against raw flattened EnCodec
        mse_loss = F.mse_loss(output, audio_flat)

        # Contrastive loss (symmetric InfoNCE)
        logit_scale = self.model.log_logit_scale.exp().clamp(
            max=self.model.max_logit_scale_val
        )
        audio_flat_norm = F.normalize(audio_flat, dim=-1)
        logits = logit_scale * output @ audio_flat_norm.T
        labels = torch.arange(output.shape[0], device=output.device)
        contrastive_loss = (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.T, labels)
        ) / 2

        total_loss = self.mse_weight * mse_loss + self.contrastive_weight * contrastive_loss

        self.accelerator.backward(total_loss)

        if self.max_grad_norm > 0:
            self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )

        self.optimizer.step()
        self.scheduler.step()

        # Monitoring
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            top1_acc = (preds == labels).float().mean().item()

        self.accelerator.log(
            {
                "loss": total_loss.item(),
                "mse_loss": mse_loss.item(),
                "contrastive_loss": contrastive_loss.item(),
                "logit_scale": logit_scale.item(),
                "top1_acc": top1_acc,
                "lr": self.scheduler.get_last_lr()[0],
            }
        )

        return {"loss": total_loss, "top1_acc": top1_acc}
