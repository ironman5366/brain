# Modified from lucidrans implementation (https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/mae.py)

# Internal imports
from models.vit import EEGViT
from models.transformer import Transformer

# External imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from pydantic import BaseModel
from accelerate import Accelerator
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from huggingface_hub import PyTorchModelHubMixin

DEFAULT_MASKING_RATIO = 0.75
DEFAULT_DECODER_DEPTH: int = 1
DEFAULT_DECODER_HEADS: int = 8
DEFAULT_DECODER_DIM_HEAD: int = 64


class EEGMAEConfig(BaseModel):
    decoder_dim: int

    masking_ratio: float = DEFAULT_MASKING_RATIO
    decoder_depth: int = DEFAULT_DECODER_DEPTH
    decoder_heads: int = DEFAULT_DECODER_HEADS
    decoder_dim_head: int = DEFAULT_DECODER_DIM_HEAD


class EEGViTMAE(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        *,
        encoder: EEGViT,
        decoder_dim,
        masking_ratio=DEFAULT_MASKING_RATIO,
        decoder_depth=DEFAULT_DECODER_DEPTH,
        decoder_heads=DEFAULT_DECODER_HEADS,
        decoder_dim_head=DEFAULT_DECODER_DIM_HEAD,
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, (
            "masking ratio must be kept between 0 and 1"
        )
        self.masking_ratio = masking_ratio
        print(f"Using masking ratio {self.masking_ratio}")

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)
        self.encoder = encoder

        # Note different from reference because our temporal embedding is 2d not 3d
        num_patches, encoder_dim = encoder.temporal_embedding.shape

        # We split apart the:

        # rearannge that does the patchify
        self.to_patch = encoder.to_patch_embedding[0]
        # and the rest of the sequential which does layernorm and projection
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        sample_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            dim_head=decoder_dim_head,
            mlp_dim=decoder_dim * 4,
        )
        self.decoder_temporal_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_sample_vals = nn.Linear(decoder_dim, sample_values_per_patch)

    def forward(self, sample):
        device = sample.device

        # get patches

        patches = self.to_patch(sample)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions

        tokens = self.patch_to_emb(patches)

        if self.encoder.pool == "cls":
            # Note different from original because 2d not 3d
            tokens += self.encoder.temporal_embedding[1 : (num_patches + 1)]
        elif self.encoder.pool == "mean":
            tokens += self.encoder.temporal_embedding.to(device, dtype=tokens.dtype)

        # calcluate # of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked

        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device=device).argsort(dim=-1)
        masked_indices, unmasked_indices = (
            rand_indices[:, :num_masked],
            rand_indices[:, num_masked:],
        )

        # Get the unmasked tokens to be encoded

        batch_range = torch.arange(batch, device=device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]
        print("har")

        # get the patches to be masked for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        # attend with vision transformer

        encoded_tokens = self.encoder.transformer(tokens)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for the decoder

        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens
        unmasked_decoder_tokens = decoder_tokens + self.decoder_temporal_emb(
            unmasked_indices
        )

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = repeat(self.mask_token, "d -> b n d", b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_temporal_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder

        decoder_tokens = torch.zeros(
            batch, num_patches, self.decoder_dim, device=device
        )
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to sample values

        mask_tokens = decoded_tokens[batch_range, masked_indices]
        pred_sample_values = self.to_sample_vals(mask_tokens)

        # Calculate reconstruction loss
        recon_loss = F.mse_loss(pred_sample_values, masked_patches)
        return recon_loss

    @classmethod
    def from_config(cls, encoder: EEGViT, config: EEGMAEConfig):
        return cls(encoder=encoder, **config.model_dump())


class MAETrainer:
    def __init__(
        self,
        *,
        mae: EEGViTMAE,
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
