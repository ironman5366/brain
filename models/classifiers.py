# Builtin imports
from typing import Literal


# Internal imports
from models.transformer import Transformer

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

    tokenizer_method: Literal["linear"] | Literal["eegnet"] | Literal["flatten"] = (
        "linear"
    )

    classifier_method: Literal["linear"] | Literal["mlp"] = "linear"
    dropout_rate: float = 0.3


class EEGNetTokenizer(nn.Module):
    def __init__(self, num_channels: int, encoder_dim: int, F1: int = 16, D: int = 2):
        super().__init__()
        F2 = F1 * D

        # Temporal conv with stride to create non-overlapping "patches"
        self.temporal_conv = nn.Conv2d(
            1,
            F1,
            (1, 64),
        )
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise_conv = nn.Conv2d(
            F1, F2, (num_channels, 1), groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(F2)
        self.elu = nn.ELU()

        self.proj = nn.Linear(F2, encoder_dim)

    def forward(self, x):
        # (B, C, T) -> (B, 1, C, T)
        x = x.unsqueeze(dim=1)
        x = self.elu(self.bn1(self.temporal_conv(x)))
        x = self.elu(self.bn2(self.depthwise_conv(x)))
        # x: (B, F2, 1, num_tokens)
        x = x.squeeze(2).permute(0, 2, 1)  # (B, num_tokens, F2)
        return self.proj(x)  # (B, num_tokens, encoder_dim)


class FlattenTokenizer(nn.Module):
    def __init__(self, encoder_dim: int):
        super().__init__()
        self.proj = nn.Linear(1, encoder_dim)

    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        # Flatten to (B, C*T, 1)
        x = x.reshape(B, C * T, 1)
        return self.proj(x)  # (B, C*T, encoder_dim)


class EEGClassifier(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        *,
        encoder_dim: int,
        heads: int,
        sequence_len: int,
        max_tokens: int,
        num_classes: int,
        tokenizer_method: str,
        classifier_method: str,
        dropout_rate: float,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.tokenizer_method = tokenizer_method

        # Which position of the time sequence we're in.
        self.positional_embedding = nn.Embedding(max_tokens, encoder_dim)

        if self.tokenizer_method == "linear":
            self.tokenizer = nn.Linear(sequence_len, encoder_dim)
        elif self.tokenizer_method == "eegnet":
            self.tokenizer = EEGNetTokenizer(
                num_channels=sequence_len, encoder_dim=encoder_dim
            )
        elif self.tokenizer_method == "flatten":
            self.tokenizer = FlattenTokenizer(encoder_dim=encoder_dim)
        else:
            raise ValueError()

        self.cls_token = nn.Parameter(torch.randn(1, encoder_dim))
        self.encoder = Transformer(
            dim=encoder_dim, heads=heads, mlp_dim=encoder_dim * 4, depth=6
        )

        if classifier_method == "linear":
            self.classifier = nn.Linear(
                in_features=encoder_dim, out_features=num_classes
            )
        elif classifier_method == "mlp":
            self.classifier = nn.Sequential(
                nn.Linear(encoder_dim, encoder_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(encoder_dim * 2, encoder_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(encoder_dim, num_classes),
            )
        else:
            raise ValueError()

    def forward(self, x):
        x = x.float()
        # X is [B, Channels, Values]. We want each input to the transformer to be a timeslice along all channels
        batch, channels, values = x.shape

        # So we permute to [B, Values, Channels], and then project channels -> dim to get [Values] tokens
        if self.tokenizer_method == "linear":
            x = x.permute(0, 2, 1)

        # tokens = self.seq_to_enc(x)
        tokens = self.tokenizer(x)
        cls_tokens = self.cls_token.expand(batch, -1, -1)

        tokens = torch.cat([cls_tokens, tokens], dim=1)

        positions = torch.arange(tokens.shape[1], device=tokens.device)

        pos_embeds = self.positional_embedding(positions)
        tokens = tokens + pos_embeds
        encoded_features = self.encoder(tokens)
        cls_output = encoded_features[:, 0]

        predictions = self.classifier(cls_output)
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

        self.accelerator.log(
            {"loss": loss, "lr": self.scheduler.get_last_lr()[0], "accuracy": accuracy}
        )

        return {
            "loss": loss,
            "num_correct": num_correct,
            "total": total,
            "accuracy": accuracy,
        }
