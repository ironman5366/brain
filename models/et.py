# Builtin imports

# Internal imports
from models.transformer import Transformer
from constants import NUM_CHANNELS

# External imports
import torch
from torch import nn
from pydantic import BaseModel
from huggingface_hub import PyTorchModelHubMixin
from einops import repeat
import torch.nn.functional as F
from accelerate import Accelerator
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler

DEFAULT_MASKING_RATIO = 0.5
DEFAULT_MAX_TOKENS = 1024


class EEGMAEConfig(BaseModel):
    encoder_dim: int
    decoder_dim: int

    heads: int
    depth: int
    dim_head: int
    mlp_dim: int

    channels: int = NUM_CHANNELS
    masking_ratio: float = DEFAULT_MASKING_RATIO
    max_tokens: int = DEFAULT_MAX_TOKENS


class EEGMAE(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        *,
        encoder_dim: int,
        decoder_dim: int,
        heads: int,
        depth: int,
        mlp_dim: int,
        channels: int,
        masking_ratio: float,
        max_tokens: int,
        dim_head: int,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim

        # What portion of the tokens will we mask out for the encoder?
        self.masking_ratio = masking_ratio

        # This is the fill-in-the-blank token that'll be repeated for all of the masked gaps
        self.mask_token = nn.Parameter(torch.randn(encoder_dim))

        # What position in the sequence we're in (which corresponds to what time of the sample)
        # TODO: replace with something rope-like for variable sequence length
        # TODO: should I have a separate embedding here for the decoder?
        self.temporal_embedding = nn.Embedding(max_tokens, encoder_dim)

        # Which channel are we in?
        self.channel_embedding = nn.Embedding(channels, encoder_dim)

        # The big boys
        self.samples_to_enc = nn.Linear(channels, encoder_dim)
        self.encoder = Transformer(
            dim=encoder_dim,
            heads=heads,
            depth=depth,
            mlp_dim=mlp_dim,
            dim_head=dim_head,
        )

        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim)
        self.decoder = Transformer(
            dim=decoder_dim,
            heads=heads,
            depth=depth,
            mlp_dim=mlp_dim,
            dim_head=dim_head,
        )
        self.dec_to_samples = nn.Linear(decoder_dim, channels)

    def forward(self, x):
        # X is [B, Channels, Values].

        # Token = a value from multiple channels at a single time. So we permute to [B, Values, Channels], and then project channels -> dim to get [Values] tokens
        rearranged = x.permute(0, 2, 1)
        tokens = self.samples_to_enc(rearranged)
        batch, num_tokens, _dim = tokens.shape

        num_masked = int(self.masking_ratio * num_tokens)
        # print(f"Masking {num_masked:,} / {num_tokens:,} tokens")

        rand_indices = torch.rand(batch, num_tokens, device=x.device).argsort(dim=-1)

        masked_indices, unmasked_indices = (
            rand_indices[:, :num_masked],
            rand_indices[:, num_masked:],
        )

        batch_idx = torch.arange(batch, device=x.device).unsqueeze(1)  # [B, 1]
        unmasked_tokens = tokens[batch_idx, unmasked_indices]  # [B, num_masked, 1024]

        # print(f"Unmasked tokens shape {unmasked_tokens.shape}")

        # Run the unmasked tokens through the encoder
        encoded_unmasked_tokens = unmasked_tokens + self.temporal_embedding(
            unmasked_indices
        ).to(unmasked_tokens.device)
        unmasked_features = self.encoder(encoded_unmasked_tokens)

        # Repeat the fill-in-the-blank mask token for the number of masked, and add the temporal embedding
        mask_tokens = repeat(self.mask_token, "d -> b n d", b=batch, n=num_masked)

        # Recombine the fill-in-the-blank tokens and the encoded features for the decode pass
        combined_features = torch.zeros(
            batch, num_tokens, self.encoder_dim, device=x.device
        )
        combined_features[batch_idx, unmasked_indices] = unmasked_features
        combined_features[batch_idx, masked_indices] = (
            mask_tokens
            + self.temporal_embedding(
                masked_indices
            )  # add the temporal embedding to the mask tokens, which weren't included when we did it earlier
        )

        # Project down to the decoder dim
        decoder_tokens = self.enc_to_dec(combined_features)

        # Decode pass
        decoded_tokens = self.decoder(decoder_tokens)

        # Project back down to the normal channel dimension
        decoded_samples = self.dec_to_samples(decoded_tokens)

        loss = F.mse_loss(decoded_samples, rearranged)
        return loss

    @classmethod
    def from_config(cls, config: EEGMAEConfig):
        return cls(**config.model_dump())


class MAETrainer:
    def __init__(
        self,
        *,
        mae: EEGMAE,
        accelerator: Accelerator,
        scheduler: LRScheduler,
        optimizer: Optimizer,
    ):
        self.mae = mae
        self.accelerator = accelerator
        self.scheduler = scheduler
        self.optimizer = optimizer

    def step(self, x):
        self.optimizer.zero_grad()
        loss = self.mae(x)

        self.accelerator.backward(loss)
        self.optimizer.step()
        self.scheduler.step()

        self.accelerator.log({"loss": loss, "lr": self.scheduler.get_last_lr()[0]})

        return {"loss": loss}
