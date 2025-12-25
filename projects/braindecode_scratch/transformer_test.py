import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from moabb.datasets import BNCI2014001
from moabb.evaluations import WithinSessionEvaluation, CrossSessionEvaluation
from moabb.paradigms import MotorImagery
from models.classifiers import EEGClassifier


# Wrap with skorch
model = NeuralNetClassifier(
    EEGClassifier,
    module__encoder_dim=1024,
    module__heads=8,
    module__sequence_len=22,
    module__num_classes=4,
    module__max_tokens=2048,
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

results = evaluation.process({"transformer": model})
print(results)
