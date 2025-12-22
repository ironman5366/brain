# Builtin imports

# Internal imports
from models.transformer import AttentionPooling

# External imports
from torch import nn
from pydantic import BaseModel
from huggingface_hub import PyTorchModelHubMixin
from accelerate import Accelerator
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torch


class EEGClassifierConfig(BaseModel):
    encoder_dim: int

    heads: int

    sequence_len: int
    max_tokens: int

    num_classes: int


class EEGClassifier(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        *,
        encoder_dim: int,
        heads: int,
        sequence_len: int,
        max_tokens: int,
        num_classes: int,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim

        # Which position of the sequence we're in - if we mask on channels corresponds to the channel index, if samples, corresponds to the timestep
        # TODO: replace with something rope-like for variable sequence length
        # TODO: should I have a separate embedding here for the decoder?
        self.positional_embedding = nn.Embedding(max_tokens, encoder_dim)

        self.seq_to_enc = nn.Linear(sequence_len, encoder_dim)

        self.encoder = AttentionPooling(
            dim=encoder_dim,
            num_heads=heads,
        )
        self.classifier = nn.Linear(in_features=encoder_dim, out_features=num_classes)

    def forward(self, x, return_debug: bool = False):
        # X is [B, Channels, Values].
        # So we permute to [B, Values, Channels], and then project channels -> dim to get [Values] tokens
        x = x.permute(0, 2, 1)

        tokens = self.seq_to_enc(x)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        # print(f"tokens shape {tokens.shape}")
        # print("positions shape", positions.shape)
        pos_embeds = self.positional_embedding(positions)
        # print("pos embeddings shape", pos_embeds.shape)
        tokens = tokens + pos_embeds
        pooled_features = self.encoder(tokens)
        # print(f"Pooled features shape {pooled_features.shape}")
        predictions = self.classifier(pooled_features)
        return predictions

    @classmethod
    def from_config(cls, config: EEGClassifierConfig):
        return cls(**config.model_dump())


class EEGClassifierTrainer:
    def __init__(
        self,
        *,
        classifier: EEGClassifier,
        accelerator: Accelerator,
        scheduler: LRScheduler,
        optimizer: Optimizer,
    ):
        self.classifier = classifier
        self.accelerator = accelerator
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()

    def step(self, x, y):
        self.optimizer.zero_grad()

        logits = self.classifier(x)
        loss = self.criterion(logits, y)

        self.accelerator.backward(loss)
        self.optimizer.step()
        self.scheduler.step()

        # Compute accuracy
        predictions = logits.argmax(dim=-1)
        num_correct = (predictions == y).sum().item()
        total = y.shape[0]
        accuracy = num_correct / total

        self.accelerator.log({"loss": loss, "lr": self.scheduler.get_last_lr()[0], "accuracy": accuracy})

        return {"loss": loss, "num_correct": num_correct, "total": total, "accuracy": accuracy}
