"""
Dumb simple classifier for use in evals
"""

# External imports
from torch import nn
from pydantic import BaseModel
from torch.optim.optimizer import Optimizer


class LinearClassifierConfig(BaseModel):
    input_dim: int
    num_classes: int


class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.layer = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.layer(x)

    @classmethod
    def from_config(cls, config: LinearClassifierConfig):
        return cls(**config.model_dump())


class LinearClassifierTrainer:
    def __init__(
        self,
        *,
        classifier: LinearClassifier,
        optimizer: Optimizer,
    ):
        self.classifier = classifier
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()

    def step(self, x, y):
        self.optimizer.zero_grad()

        logits = self.classifier(x)
        loss = self.criterion(logits, y)
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}
