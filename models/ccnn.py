# From torcheeg
from typing import Tuple

import torch
import torch.nn as nn
from pydantic import BaseModel
from huggingface_hub import PyTorchModelHubMixin
from accelerate import Accelerator
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class CCNN(nn.Module):
    r"""
    Continuous Convolutional Neural Network (CCNN). For more details, please refer to the following information.

    - Paper: Yang Y, Wu Q, Fu Y, et al. Continuous convolutional neural network with 3D input for EEG-based emotion recognition[C]//International Conference on Neural Information Processing. Springer, Cham, 2018: 433-443.
    - URL: https://link.springer.com/chapter/10.1007/978-3-030-04239-4_39
    - Related Project: https://github.com/ynulonger/DE_CNN

    Below is a recommended suite for use in emotion recognition tasks:

    .. code-block:: python

        from torcheeg.datasets import DEAPDataset
        from torcheeg import transforms
        from torcheeg.datasets.constants import DEAP_CHANNEL_LOCATION_DICT
        from torcheeg.models import CCNN
        from torch.utils.data import DataLoader

        dataset = DEAPDataset(root_path='./data_preprocessed_python',
                              offline_transform=transforms.Compose([
                                  transforms.BandDifferentialEntropy(),
                                  transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
                              ]),
                              online_transform=transforms.ToTensor(),
                              label_transform=transforms.Compose([
                                  transforms.Select('valence'),
                                  transforms.Binary(5.0),
                              ]))

        model = CCNN(num_classes=2, in_channels=4, grid_size=(9, 9))

        x, y = next(iter(DataLoader(dataset, batch_size=64)))
        model(x)

    Args:
        in_channels (int): The feature dimension of each electrode. (default: :obj:`4`)
        grid_size (tuple): Spatial dimensions of grid-like EEG representation. (default: :obj:`(9, 9)`)
        num_classes (int): The number of classes to predict. (default: :obj:`2`)
        dropout (float): Probability of an element to be zeroed in the dropout layers. (default: :obj:`0.25`)
    """

    def __init__(
        self,
        in_channels: int = 4,
        grid_size: Tuple[int, int] = (9, 9),
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super(CCNN, self).__init__()
        self.in_channels = in_channels
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.dropout = dropout

        self.conv1 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(self.in_channels, 64, kernel_size=4, stride=1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(64, 128, kernel_size=4, stride=1),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(128, 256, kernel_size=4, stride=1),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(256, 64, kernel_size=4, stride=1),
            nn.ReLU(),
        )

        self.lin1 = nn.Sequential(
            nn.Linear(self.grid_size[0] * self.grid_size[1] * 64, 1024),
            nn.SELU(),  # Not mentioned in paper
            nn.Dropout2d(self.dropout),
        )
        self.lin2 = nn.Linear(1024, self.num_classes)

    def feature_dim(self):
        return self.grid_size[0] * self.grid_size[1] * 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 4, 9, 9]`. Here, :obj:`n` corresponds to the batch size, :obj:`4` corresponds to :obj:`in_channels`, and :obj:`(9, 9)` corresponds to :obj:`grid_size`.

        Returns:
            torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.flatten(start_dim=1)

        x = self.lin1(x)
        x = self.lin2(x)
        return x


# ============================================================================
# 1D CCNN Variant for raw EEG regression
# ============================================================================


class CCNN1DConfig(BaseModel):
    """Configuration for 1D CCNN regressor."""

    in_channels: int = 32  # Number of EEG channels
    seq_len: int = 128  # Sequence length (samples)
    dropout: float = 0.5
    base_channels: int = 64  # Base channel count for conv layers
    hidden_dim: int = 1024  # Hidden dimension for linear layer


class CCNN1DRegressor(nn.Module, PyTorchModelHubMixin):
    """
    1D variant of CCNN for regression on raw EEG data.

    Adapted from the 2D CCNN to work with 1D sequences instead of 2D grids.
    Input shape: (batch, in_channels, seq_len) e.g., (B, 32, 128)
    Output: single continuous value for regression

    Args:
        in_channels: Number of input channels (EEG electrodes). Default: 32
        seq_len: Sequence length (time samples). Default: 128
        dropout: Dropout probability. Default: 0.5
        base_channels: Base channel count for conv layers. Default: 64
        hidden_dim: Hidden dimension for linear layer. Default: 1024
    """

    def __init__(
        self,
        in_channels: int = 32,
        seq_len: int = 128,
        dropout: float = 0.5,
        base_channels: int = 64,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.dropout = dropout
        self.base_channels = base_channels
        self.hidden_dim = hidden_dim

        # 1D convolution layers (matching CCNN structure)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(base_channels * 4, base_channels, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
        )

        # Calculate output size after convolutions
        # With padding=2 and kernel=4, stride=1: output_len = input_len + 1
        # After 4 conv layers: seq_len + 4
        conv_out_len = seq_len + 4
        self.flat_features = base_channels * conv_out_len

        self.lin1 = nn.Sequential(
            nn.Linear(self.flat_features, hidden_dim),
            nn.SELU(),
            nn.Dropout(dropout),
        )
        self.lin2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: EEG signal of shape (batch, in_channels, seq_len)

        Returns:
            Regression output of shape (batch,)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.flatten(start_dim=1)

        x = self.lin1(x)
        x = self.lin2(x)
        return x.squeeze(-1)  # (batch,)

    @classmethod
    def from_config(cls, config: CCNN1DConfig):
        return cls(**config.model_dump())


class CCNN1DRegressorTrainer:
    """Trainer for CCNN1D regression model."""

    def __init__(
        self,
        *,
        model: CCNN1DRegressor,
        accelerator: Accelerator,
        scheduler: LRScheduler,
        optimizer: Optimizer,
    ):
        self.model = model
        self.accelerator = accelerator
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.criterion = nn.MSELoss()

    def step(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        """
        Single training step.

        Args:
            x: Input tensor (batch, channels, seq_len)
            y: Target tensor (batch,)

        Returns:
            Dict with loss and MAE metrics
        """
        self.optimizer.zero_grad()

        predictions = self.model(x)
        loss = self.criterion(predictions, y)

        self.accelerator.backward(loss)
        self.optimizer.step()
        self.scheduler.step()

        # Compute metrics
        mae = torch.abs(predictions - y).mean().item()

        # Accuracy within 1 point (on 0-9 scale)
        within_1 = (torch.abs(predictions - y) <= 1.0).float().mean().item()

        # Binary accuracy (threshold at 5.0 for high/low)
        pred_high = predictions >= 5.0
        target_high = y >= 5.0
        binary_acc = (pred_high == target_high).float().mean().item()

        self.accelerator.log({
            "loss": loss.item(),
            "mae": mae,
            "within_1": within_1,
            "binary_acc": binary_acc,
            "lr": self.scheduler.get_last_lr()[0],
        })

        return {
            "loss": loss.item(),
            "mae": mae,
            "within_1": within_1,
            "binary_acc": binary_acc,
        }


# ============================================================================
# 1D CCNN Variant for classification
# ============================================================================


class CCNN1DClassifierConfig(BaseModel):
    """Configuration for 1D CCNN classifier."""

    in_channels: int = 32  # Number of EEG channels
    seq_len: int = 128  # Sequence length (samples)
    num_classes: int = 2  # Number of output classes
    dropout: float = 0.5
    base_channels: int = 64  # Base channel count for conv layers
    hidden_dim: int = 1024  # Hidden dimension for linear layer


class CCNN1DClassifier(nn.Module, PyTorchModelHubMixin):
    """
    1D variant of CCNN for classification on raw EEG data.

    Input shape: (batch, in_channels, seq_len) e.g., (B, 32, 128)
    Output: class logits of shape (batch, num_classes)

    Args:
        in_channels: Number of input channels (EEG electrodes). Default: 32
        seq_len: Sequence length (time samples). Default: 128
        num_classes: Number of output classes. Default: 2
        dropout: Dropout probability. Default: 0.5
        base_channels: Base channel count for conv layers. Default: 64
        hidden_dim: Hidden dimension for linear layer. Default: 1024
    """

    def __init__(
        self,
        in_channels: int = 32,
        seq_len: int = 128,
        num_classes: int = 2,
        dropout: float = 0.5,
        base_channels: int = 64,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.dropout = dropout
        self.base_channels = base_channels
        self.hidden_dim = hidden_dim

        # 1D convolution layers (matching CCNN structure)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(base_channels * 4, base_channels, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
        )

        # Calculate output size after convolutions
        conv_out_len = seq_len + 4
        self.flat_features = base_channels * conv_out_len

        self.lin1 = nn.Sequential(
            nn.Linear(self.flat_features, hidden_dim),
            nn.SELU(),
            nn.Dropout(dropout),
        )
        self.lin2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: EEG signal of shape (batch, in_channels, seq_len)

        Returns:
            Class logits of shape (batch, num_classes)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.flatten(start_dim=1)

        x = self.lin1(x)
        x = self.lin2(x)
        return x

    @classmethod
    def from_config(cls, config: CCNN1DClassifierConfig):
        return cls(**config.model_dump())


class CCNN1DClassifierTrainer:
    """Trainer for CCNN1D classification model."""

    def __init__(
        self,
        *,
        model: CCNN1DClassifier,
        accelerator: Accelerator,
        scheduler: LRScheduler,
        optimizer: Optimizer,
    ):
        self.model = model
        self.accelerator = accelerator
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()

    def step(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        """
        Single training step.

        Args:
            x: Input tensor (batch, channels, seq_len)
            y: Target tensor (batch,) with class indices

        Returns:
            Dict with loss and accuracy metrics
        """
        self.optimizer.zero_grad()

        logits = self.model(x)
        loss = self.criterion(logits, y)

        self.accelerator.backward(loss)
        self.optimizer.step()
        self.scheduler.step()

        # Compute accuracy
        predictions = logits.argmax(dim=-1)
        num_correct = (predictions == y).sum().item()
        total = y.shape[0]
        accuracy = num_correct / total

        self.accelerator.log({
            "loss": loss.item(),
            "accuracy": accuracy,
            "lr": self.scheduler.get_last_lr()[0],
        })

        return {
            "loss": loss.item(),
            "accuracy": accuracy,
            "num_correct": num_correct,
            "total": total,
        }
