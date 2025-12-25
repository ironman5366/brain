# External imports
from torch import nn
from pydantic import BaseModel
from accelerate import Accelerator
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from huggingface_hub import PyTorchModelHubMixin


class EEGNetConfig(BaseModel):
    num_channels: int
    num_samples: int
    num_classes: int

    # The number of temporal filters
    F1: int = 8

    # The number of pointwise filters
    F2: int = 16
    # The depth of the depthwise convolution
    D: int = 2

    dropout_rate: float = 0.5


class EEGNet(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        *,
        num_channels: int,
        num_samples: int,
        num_classes: int,
        F1: int,
        F2: int,
        D: int,
        dropout_rate: float,
    ):
        super().__init__()
        # Block 1

        # Temporal convolution
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=F1, kernel_size=(1, 64), padding="same"
        )
        self.norm1 = nn.BatchNorm2d(F1)

        # DepthwiseConv2D
        self.depthwise = nn.Conv2d(
            in_channels=F1,
            out_channels=D * F1,
            kernel_size=(num_channels, 1),
            groups=F1,  # this makes it depthwise
        )
        self.norm2 = nn.BatchNorm2d(D * F1)
        self.activation1 = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # Block 2
        self.separable_depthwise = nn.Conv2d(
            D * F1, D * F1, kernel_size=(1, 16), padding="same", groups=D * F1
        )
        self.separable_pointwise = nn.Conv2d(D * F1, F2, kernel_size=(1, 1))
        self.norm3 = nn.BatchNorm2d(F2)
        self.activation2 = nn.ELU()
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)

        # Classifier
        self.flatten = nn.Flatten()
        # Output size after pooling: F2 * (T // 32)
        self.dense = nn.Linear(F2 * (num_samples // 32), num_classes)

    def forward(self, x):
        x = x.float()
        # X of shape [B, Channels, Values]
        # Start by reshaping to (B, 1, C, V)
        x = x.unsqueeze(dim=1)

        # Block 1
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.depthwise(x)
        x = self.norm2(x)
        x = self.activation1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.separable_depthwise(x)
        x = self.separable_pointwise(x)
        x = self.norm3(x)
        x = self.activation2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Classifier
        x = self.flatten(x)
        x = self.dense(x)
        return x

    @classmethod
    def from_config(cls, config: EEGNetConfig):
        return cls(**config.model_dump())


class EEGNetTrainer:
    def __init__(
        self,
        *,
        classifier: EEGNet,
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
