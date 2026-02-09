"""
Hierarchical EEG → Audio Contrastive model for continuous/multi-second sequences.

CLIP-style contrastive learning with hierarchical encoding:
  Level 1 (Spatial): Per-second channel transformer for EEG, per-second frame transformer for audio.
  Level 2 (Temporal): Transformer across seconds for both modalities.

Projects both to a shared normalized embedding space with symmetric InfoNCE loss.
Optionally includes per-step MSE reconstruction head.
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


class HierarchicalEEGAudioContrastiveConfig(BaseModel):
    # Level 1: EEG spatial encoder
    encoder_dim: int = 512
    encoder_mlp_dim: int = 2048
    heads: int = 8
    depth: int = 6
    dim_head: int = 64
    dropout: float = 0.1
    num_channels: int = 120
    eeg_samples: int = 125

    # Level 1: Audio per-second encoder
    audio_embed_dim: int = 128
    audio_frames: int = 75
    audio_encoder_depth: int = 4
    audio_encoder_heads: int = 8
    audio_encoder_dim_head: int = 64
    audio_encoder_mlp_dim: int = 2048

    # Level 2: Temporal encoder (shared config for both modalities)
    temporal_dim: int = 512
    temporal_depth: int = 4
    temporal_heads: int = 8
    temporal_dim_head: int = 64
    temporal_mlp_dim: int = 2048
    max_seq_len: int = 300

    # Contrastive learning
    shared_embed_dim: int = 256
    init_logit_scale: float = 2.6593
    max_logit_scale: float = 4.6052

    # Loss weights
    contrastive_weight: float = 1.0
    mse_weight: float = 0.5


class HierarchicalEEGAudioContrastive(nn.Module, PyTorchModelHubMixin):
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
        temporal_dim: int,
        temporal_depth: int,
        temporal_heads: int,
        temporal_dim_head: int,
        temporal_mlp_dim: int,
        max_seq_len: int,
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
        self.encoder_dim = encoder_dim

        # --- EEG Level 1: Spatial encoder (per 1-second chunk) ---
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

        # --- EEG Level 2: Temporal encoder ---
        self.eeg_temporal_project = nn.Linear(encoder_dim, temporal_dim)
        self.eeg_temporal_pos = nn.Embedding(max_seq_len, temporal_dim)
        self.eeg_temporal_encoder = Transformer(
            dim=temporal_dim,
            depth=temporal_depth,
            heads=temporal_heads,
            mlp_dim=temporal_mlp_dim,
            dim_head=temporal_dim_head,
            dropout=dropout,
        )

        # --- EEG projector to shared space ---
        self.eeg_projector = nn.Sequential(
            nn.LayerNorm(temporal_dim),
            nn.Linear(temporal_dim, temporal_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_dim, shared_embed_dim),
        )

        # --- Audio Level 1: Per-second frame encoder ---
        self.frame_embed = nn.Linear(audio_embed_dim, encoder_dim)
        self.audio_pos_embed = nn.Embedding(audio_frames, encoder_dim)
        self.audio_spatial_encoder = Transformer(
            dim=encoder_dim,
            depth=audio_encoder_depth,
            heads=audio_encoder_heads,
            mlp_dim=audio_encoder_mlp_dim,
            dim_head=audio_encoder_dim_head,
            dropout=dropout,
        )

        # --- Audio Level 2: Temporal encoder ---
        self.audio_temporal_project = nn.Linear(encoder_dim, temporal_dim)
        self.audio_temporal_pos = nn.Embedding(max_seq_len, temporal_dim)
        self.audio_temporal_encoder = Transformer(
            dim=temporal_dim,
            depth=temporal_depth,
            heads=temporal_heads,
            mlp_dim=temporal_mlp_dim,
            dim_head=temporal_dim_head,
            dropout=dropout,
        )

        # --- Audio projector to shared space ---
        self.audio_projector = nn.Sequential(
            nn.LayerNorm(temporal_dim),
            nn.Linear(temporal_dim, temporal_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_dim, shared_embed_dim),
        )

        # --- Learnable temperature ---
        self.log_logit_scale = nn.Parameter(torch.tensor(init_logit_scale))

        # --- Optional MSE reconstruction head (per-step) ---
        if mse_weight > 0:
            output_dim = audio_embed_dim * audio_frames
            self.mse_head = nn.Sequential(
                nn.LayerNorm(temporal_dim),
                nn.Linear(temporal_dim, temporal_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(temporal_dim, output_dim),
            )
        else:
            self.mse_head = None

    def _build_padding_mask(
        self, W: int, lengths: torch.Tensor | None, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor | None:
        """Build additive attention mask for variable-length sequences."""
        if lengths is None:
            return None
        B = lengths.shape[0]
        valid = torch.arange(W, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        attn_mask = valid.unsqueeze(1).unsqueeze(2).expand(B, 1, W, W).float()
        attn_mask = (1.0 - attn_mask) * torch.finfo(dtype).min
        return attn_mask

    def encode_eeg(
        self, eeg: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode multi-second EEG sequence.

        Args:
            eeg: (B, W, C, T)
            lengths: (B,) optional

        Returns:
            eeg_embed: (B, shared_embed_dim) L2-normalized
            eeg_temporal: (B, W, temporal_dim) temporal features (for MSE head)
        """
        B, W, C, T = eeg.shape

        # Level 1: Spatial encode each chunk
        eeg_flat = eeg.reshape(B * W, C, T)
        tokens = self.channel_embed(eeg_flat)
        positions = torch.arange(C, device=eeg.device)
        tokens = tokens + self.pos_embed(positions)
        encoded = self.spatial_encoder(tokens)
        spatial_embeds = encoded.mean(dim=1).reshape(B, W, -1)  # (B, W, encoder_dim)

        # Level 2: Temporal encode
        temporal_input = self.eeg_temporal_project(spatial_embeds)
        t_positions = torch.arange(W, device=eeg.device)
        temporal_input = temporal_input + self.eeg_temporal_pos(t_positions)

        attn_mask = self._build_padding_mask(W, lengths, eeg.device, temporal_input.dtype)
        temporal_encoded = self.eeg_temporal_encoder(temporal_input, attn_mask=attn_mask)

        # Pool: mean over valid positions
        if lengths is not None:
            valid_mask = torch.arange(W, device=eeg.device).unsqueeze(0) < lengths.unsqueeze(1)
            valid_mask = valid_mask.unsqueeze(-1).float()
            pooled = (temporal_encoded * valid_mask).sum(dim=1) / valid_mask.sum(dim=1)
        else:
            pooled = temporal_encoded.mean(dim=1)

        eeg_embed = F.normalize(self.eeg_projector(pooled), dim=-1)
        return eeg_embed, temporal_encoded

    def encode_audio(
        self, audio: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Encode multi-second audio sequence.

        Args:
            audio: (B, W, audio_embed_dim, audio_frames)
            lengths: (B,) optional

        Returns:
            audio_embed: (B, shared_embed_dim) L2-normalized
        """
        B, W, A_dim, A_frames = audio.shape

        # Level 1: Per-second frame encode
        audio_flat = audio.reshape(B * W, A_dim, A_frames)
        audio_t = audio_flat.transpose(1, 2)  # (B*W, audio_frames, audio_embed_dim)
        frame_tokens = self.frame_embed(audio_t)  # (B*W, audio_frames, encoder_dim)
        a_positions = torch.arange(A_frames, device=audio.device)
        frame_tokens = frame_tokens + self.audio_pos_embed(a_positions)
        audio_encoded = self.audio_spatial_encoder(frame_tokens)
        audio_spatial = audio_encoded.mean(dim=1).reshape(B, W, -1)  # (B, W, encoder_dim)

        # Level 2: Temporal encode
        temporal_input = self.audio_temporal_project(audio_spatial)
        t_positions = torch.arange(W, device=audio.device)
        temporal_input = temporal_input + self.audio_temporal_pos(t_positions)

        attn_mask = self._build_padding_mask(W, lengths, audio.device, temporal_input.dtype)
        temporal_encoded = self.audio_temporal_encoder(temporal_input, attn_mask=attn_mask)

        # Pool: mean over valid positions
        if lengths is not None:
            valid_mask = torch.arange(W, device=audio.device).unsqueeze(0) < lengths.unsqueeze(1)
            valid_mask = valid_mask.unsqueeze(-1).float()
            pooled = (temporal_encoded * valid_mask).sum(dim=1) / valid_mask.sum(dim=1)
        else:
            pooled = temporal_encoded.mean(dim=1)

        audio_embed = F.normalize(self.audio_projector(pooled), dim=-1)
        return audio_embed

    def forward(
        self,
        eeg: torch.Tensor,
        audio: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Args:
            eeg: (B, W, C, T) multi-second EEG
            audio: (B, W, audio_embed_dim, audio_frames) multi-second audio
            lengths: (B,) optional actual lengths

        Returns:
            eeg_embed: (B, shared_embed_dim) L2-normalized
            audio_embed: (B, shared_embed_dim) L2-normalized
            mse_pred: (B, W, audio_embed_dim, audio_frames) or None
        """
        eeg_embed, eeg_temporal = self.encode_eeg(eeg, lengths)
        audio_embed = self.encode_audio(audio, lengths)

        # Optional MSE reconstruction from EEG temporal features
        mse_pred = None
        if self.mse_head is not None:
            B, W = eeg_temporal.shape[:2]
            mse_pred = self.mse_head(eeg_temporal)  # (B, W, embed_dim * frames)
            mse_pred = mse_pred.reshape(B, W, self.audio_embed_dim, self.audio_frames)

        return eeg_embed, audio_embed, mse_pred

    def load_spatial_from_baseline(self, baseline_checkpoint: str):
        """Initialize spatial encoder weights from a trained contrastive baseline."""
        from models.audio_contrastive import EEGAudioContrastive

        baseline = EEGAudioContrastive.from_pretrained(baseline_checkpoint)
        # EEG spatial encoder
        self.channel_embed.load_state_dict(baseline.channel_embed.state_dict())
        self.pos_embed.load_state_dict(baseline.pos_embed.state_dict())
        self.spatial_encoder.load_state_dict(baseline.encoder.state_dict())
        # Audio spatial encoder
        self.frame_embed.load_state_dict(baseline.frame_embed.state_dict())
        self.audio_pos_embed.load_state_dict(baseline.audio_pos_embed.state_dict())
        self.audio_spatial_encoder.load_state_dict(baseline.audio_encoder.state_dict())
        print(f"Loaded spatial encoders from {baseline_checkpoint}")

    @classmethod
    def from_config(cls, config: HierarchicalEEGAudioContrastiveConfig):
        return cls(**config.model_dump())


class HierarchicalEEGAudioContrastiveTrainer:
    def __init__(
        self,
        *,
        model: HierarchicalEEGAudioContrastive,
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

    def step(
        self,
        eeg: torch.Tensor,
        audio_embeds: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> dict:
        self.optimizer.zero_grad()

        eeg_embed, audio_embed, mse_pred = self.model(eeg, audio_embeds, lengths)

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
            if lengths is not None:
                B, W = mse_pred.shape[:2]
                mask = torch.arange(W, device=eeg.device).unsqueeze(0) < lengths.unsqueeze(1)
                mask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(mse_pred)
                mse_loss = F.mse_loss(mse_pred[mask], audio_embeds[mask])
            else:
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

        # In-batch retrieval accuracy
        with torch.no_grad():
            preds = logits_eeg.argmax(dim=-1)
            top1_acc = (preds == labels).float().mean().item()

        self.accelerator.log({
            "loss": total_loss.item(),
            "contrastive_loss": contrastive_loss.item(),
            "mse_loss": mse_loss_val,
            "logit_scale": logit_scale.item(),
            "top1_acc": top1_acc,
            "lr": self.scheduler.get_last_lr()[0],
        })

        return {"loss": total_loss, "top1_acc": top1_acc}
