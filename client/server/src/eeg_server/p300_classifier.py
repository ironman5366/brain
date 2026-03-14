"""P300 binary classifier — EEGNet trained on copy-spelling calibration data."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .timing import estimate_marker_offset_seconds, marker_epoch_seconds

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - depends on local env
    torch = None

    class _ModulePlaceholder:
        def __init__(self, *args, **kwargs):
            pass

    class _NNPlaceholder:
        Module = _ModulePlaceholder

    nn = _NNPlaceholder()

logger = logging.getLogger(__name__)

# --- Model ---

POST_STIMULUS_MS = 800
SAMPLE_RATE = 250
POST_STIMULUS_SAMPLES = int(POST_STIMULUS_MS * SAMPLE_RATE / 1000)  # 200
PRE_STIMULUS_MS = 200
PRE_STIMULUS_SAMPLES = int(PRE_STIMULUS_MS * SAMPLE_RATE / 1000)  # 50


class P300EEGNet(nn.Module):
    """Compact EEGNet for binary P300 classification.

    Input:  (batch, n_channels, 200)  — 800ms post-stimulus at 250Hz
    Output: (batch, 2)                — logits for [non-target, target]
    """

    def __init__(
        self,
        n_channels: int = 8,
        n_samples: int = POST_STIMULUS_SAMPLES,
        F1: int = 8,
        F2: int = 16,
        D: int = 2,
        dropout: float = 0.5,
    ):
        _require_torch()
        super().__init__()
        self.config = {
            "n_channels": n_channels,
            "n_samples": n_samples,
            "F1": F1,
            "F2": F2,
            "D": D,
            "dropout": dropout,
        }

        # Block 1: temporal + spatial filtering
        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, 32), padding="same")
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(
            F1, D * F1, kernel_size=(n_channels, 1), groups=F1
        )
        self.bn2 = nn.BatchNorm2d(D * F1)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Block 2: separable convolution
        self.sep_depth = nn.Conv2d(
            D * F1, D * F1, kernel_size=(1, 16), padding="same", groups=D * F1
        )
        self.sep_point = nn.Conv2d(D * F1, F2, kernel_size=(1, 1))
        self.bn3 = nn.BatchNorm2d(F2)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        # Classifier
        self.flatten = nn.Flatten()
        flat_size = F2 * (n_samples // 32)
        self.dense = nn.Linear(flat_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, T) -> (B, 1, C, T)
        x = x.unsqueeze(1)
        x = self.bn1(self.conv1(x))
        x = self.drop1(self.pool1(self.act1(self.bn2(self.depthwise(x)))))
        x = self.sep_point(self.sep_depth(x))
        x = self.drop2(self.pool2(self.act2(self.bn3(x))))
        return self.dense(self.flatten(x))


# --- Preprocessing ---


def preprocess_eeg(eeg: np.ndarray, sr: int) -> np.ndarray:
    """Apply CAR, bandpass, and notch filter to raw EEG.

    Args:
        eeg: (n_channels, n_samples) raw EEG data
        sr: sampling rate in Hz

    Returns:
        Filtered EEG array of same shape.
    """
    # Common Average Reference
    eeg = eeg - eeg.mean(axis=0, keepdims=True)

    # Bandpass 0.5-15 Hz
    nyq = sr / 2
    b, a = butter(4, [0.5 / nyq, 15.0 / nyq], btype="band")
    eeg = filtfilt(b, a, eeg, axis=1)

    # 60 Hz notch
    b_notch, a_notch = iirnotch(60.0, 30.0, sr)
    eeg = filtfilt(b_notch, a_notch, eeg, axis=1)

    return eeg


def extract_labeled_epochs(
    eeg: np.ndarray,
    timestamps: np.ndarray,
    markers: list[dict],
    sr: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract labeled post-stimulus epochs from a copy-spelling session.

    Prefers client-provided epoch timestamps for precise alignment. Falls back
    to performance.now() + a derived server offset for older sessions.

    Returns:
        (epochs, labels) where epochs is (N, n_channels, 200) and labels is (N,).
    """
    offset = estimate_marker_offset_seconds(markers)
    if offset is None:
        raise RuntimeError("No markers with usable timing information")

    pre = PRE_STIMULUS_SAMPLES
    post = POST_STIMULUS_SAMPLES
    n_samples = eeg.shape[1]

    epochs = []
    labels = []
    n_rejected = 0

    for m in markers:
        if m.get("code") != "p300_flash":
            continue
        meta = m.get("metadata")
        if meta is None or "is_target" not in meta:
            continue

        unix_time = marker_epoch_seconds(m, offset)
        if unix_time is None:
            n_rejected += 1
            continue
        sample_idx = np.searchsorted(timestamps, unix_time)

        start = sample_idx - pre
        end = sample_idx + post

        if start < 0 or end > n_samples:
            n_rejected += 1
            continue

        epoch = eeg[:, start:end].copy()

        # Baseline correction (pre-stimulus mean)
        baseline = epoch[:, :pre].mean(axis=1, keepdims=True)
        epoch -= baseline

        # Artifact rejection
        if np.abs(epoch).max() > 150.0:
            n_rejected += 1
            continue

        # Take only post-stimulus portion for the model
        epochs.append(epoch[:, pre:])
        labels.append(1 if meta["is_target"] else 0)

    if not epochs:
        raise RuntimeError(f"No valid epochs extracted ({n_rejected} rejected)")

    logger.info(
        "Extracted %d epochs (%d target, %d non-target, %d rejected)",
        len(epochs),
        sum(labels),
        len(labels) - sum(labels),
        n_rejected,
    )

    return np.array(epochs, dtype=np.float32), np.array(labels, dtype=np.int64)


# --- Training ---


def _get_device() -> torch.device:
    _require_torch()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_p300_classifier(
    epochs: np.ndarray,
    labels: np.ndarray,
    n_channels: int = 8,
    n_epochs: int = 100,
) -> tuple[P300EEGNet, dict]:
    """Train P300EEGNet on labeled epochs.

    Args:
        epochs: (N, n_channels, 200) post-stimulus epochs
        labels: (N,) binary labels (0=non-target, 1=target)
        n_channels: number of EEG channels
        n_epochs: training epochs

    Returns:
        (trained_model, metrics_dict)
    """
    _require_torch()
    device = _get_device()
    logger.info("Training P300 classifier on %s (%d samples)", device, len(labels))

    # Train/val split (stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        epochs, labels, test_size=0.2, stratify=labels, random_state=42,
    )

    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    # Class weights for imbalanced data
    n_target = (labels == 1).sum()
    n_nontarget = (labels == 0).sum()
    weight = torch.tensor(
        [len(labels) / (2 * n_nontarget), len(labels) / (2 * n_target)],
        dtype=torch.float32,
    ).to(device)

    model = P300EEGNet(n_channels=n_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5,
    )
    criterion = nn.CrossEntropyLoss(weight=weight)

    best_auc = 0.0
    best_state = None
    batch_size = 32

    for epoch in range(n_epochs):
        # Training
        model.train()
        perm = torch.randperm(len(X_train_t))
        train_loss = 0.0
        train_correct = 0

        for i in range(0, len(X_train_t), batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(xb)
            train_correct += (logits.argmax(1) == yb).sum().item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
            val_preds = val_logits.argmax(1)
            val_acc = (val_preds == y_val_t).float().mean().item()

            val_probs = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
            try:
                val_auc = roc_auc_score(y_val, val_probs)
            except ValueError:
                val_auc = 0.0

        scheduler.step(val_loss)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            logger.info(
                "Epoch %d/%d — val_acc=%.3f val_auc=%.3f",
                epoch + 1, n_epochs, val_acc, val_auc,
            )

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    model.to("cpu")

    # Final metrics on val set
    with torch.no_grad():
        final_logits = model(torch.from_numpy(X_val))
        final_probs = torch.softmax(final_logits, dim=1)[:, 1].numpy()
        final_acc = (final_logits.argmax(1).numpy() == y_val).mean()
        try:
            final_auc = roc_auc_score(y_val, final_probs)
        except ValueError:
            final_auc = 0.0

    metrics = {
        "val_accuracy": round(float(final_acc), 3),
        "val_auc": round(float(final_auc), 3),
        "n_target": int(n_target),
        "n_nontarget": int(n_nontarget),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "best_epoch_auc": round(best_auc, 3),
    }
    logger.info("Training complete: %s", metrics)
    return model, metrics


# --- Save / Load ---


def save_model(model: P300EEGNet, path: Path, metrics: dict) -> None:
    _require_torch()
    torch.save(
        {"state_dict": model.state_dict(), "config": model.config, "metrics": metrics},
        path,
    )
    logger.info("P300 model saved to %s", path)


def load_model(path: Path) -> tuple[P300EEGNet, dict]:
    _require_torch()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = P300EEGNet(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    logger.info("P300 model loaded from %s", path)
    return model, checkpoint.get("metrics", {})


# --- Inference ---


def score_epochs(model: P300EEGNet, epochs: np.ndarray) -> np.ndarray:
    """Run epochs through the classifier and return P(target) for each.

    Args:
        model: trained P300EEGNet in eval mode
        epochs: (N, n_channels, 200) post-stimulus epochs

    Returns:
        (N,) array of target probabilities
    """
    _require_torch()
    x = torch.from_numpy(epochs.astype(np.float32))
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1]
    return probs.numpy()


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for P300 classifier support. "
            "Install torch in client/server to enable calibration and model-based scoring."
        )
