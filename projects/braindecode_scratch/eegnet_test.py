import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from moabb.datasets import BNCI2014001
from moabb.evaluations import WithinSessionEvaluation, CrossSessionEvaluation
from moabb.paradigms import MotorImagery
from models.eegnet import EEGNet


# Wrap with skorch
model = NeuralNetClassifier(
    EEGNet,
    module__num_channels=22,
    module__num_classes=4,
    module__num_samples=1001,
    module__F1=8,
    module__F2=16,
    module__D=2,
    module__dropout_rate=0.5,
    max_epochs=50,
    lr=0.001,
    batch_size=32,
    optimizer=torch.optim.Adam,
    criterion=nn.CrossEntropyLoss,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# Use with MOABB
dataset = BNCI2014001()
paradigm = MotorImagery()

evaluation = CrossSessionEvaluation(
    paradigm=paradigm,
    datasets=[dataset],
)

results = evaluation.process({"EEGNet": model})
print(results)
