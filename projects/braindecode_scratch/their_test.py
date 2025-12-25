import torch
from braindecode import EEGClassifier
from braindecode.models import ShallowFBCSPNet, EEGNetv4
from moabb.datasets import BNCI2014001
from moabb.evaluations import CrossSessionEvaluation
from moabb.paradigms import MotorImagery

# Load dataset and paradigm
dataset = BNCI2014001()
paradigm = MotorImagery()

# Create a Braindecode model wrapped in EEGClassifier (sklearn-compatible)
model = EEGClassifier(
    module=ShallowFBCSPNet,
    module__n_chans=22,  # number of EEG channels
    module__n_outputs=4,  # number of classes
    module__n_times=1001,  # number of time samples
    module__final_conv_length="auto",
    max_epochs=100,
    batch_size=64,
    optimizer=torch.optim.Adam,
    optimizer__lr=0.001,
    device="cuda",
)

# Run evaluation
evaluation = CrossSessionEvaluation(
    paradigm=paradigm,
    datasets=[dataset],
)

results = evaluation.process({"ShallowFBCSP": model})
print(results)
