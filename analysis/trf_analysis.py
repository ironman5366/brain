"""
Temporal Response Function (TRF) analysis for NMED music-listening EEG.

Tests whether the EEG data contains decodable auditory information using
classical linear methods (ridge regression with time lags).

Two analyses:
  1. Backward TRF (stimulus reconstruction): predict audio envelope from EEG
  2. Forward TRF (encoding model): predict EEG from audio envelope

Per-subject, leave-one-song-out cross-validation, with permutation significance testing.
GPU-accelerated ridge regression via PyTorch.

Usage:
    uv run python analysis/trf_analysis.py
    uv run python analysis/trf_analysis.py --n-perms 500
    uv run python analysis/trf_analysis.py --skip-permutation
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import scipy.io as sio
import soundfile as sf
import torch
from scipy.signal import butter, filtfilt, hilbert, resample

sys.path.append(str(Path(__file__).parent.parent))

from data.nmed.songs import (
    NMED_AUDIO_DIR,
    NMED_DATA_DIR,
    NMED_SFREQ,
    SONGS,
    SONG_BY_ID,
    SUBJECT_IDS,
)

RESULTS_DIR = Path("reports/trf")
SFREQ = NMED_SFREQ  # 125 Hz
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Audio envelope extraction
# ---------------------------------------------------------------------------

def extract_envelope(audio_path: Path, target_sr: int = SFREQ) -> np.ndarray:
    """Extract broadband amplitude envelope, downsampled to target_sr."""
    audio, sr = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Downsample to 1000 Hz first (fast Hilbert on ~300K samples)
    intermediate_sr = 1000
    n_intermediate = int(len(audio) * intermediate_sr / sr)
    audio_1k = resample(audio, n_intermediate).astype(np.float64)

    # Hilbert -> amplitude envelope
    envelope = np.abs(hilbert(audio_1k))

    # Low-pass at 30 Hz
    b, a = butter(4, 30.0 / (intermediate_sr / 2.0), btype="low")
    envelope = filtfilt(b, a, envelope)

    # Final resample to target_sr
    n_target = int(len(envelope) * target_sr / intermediate_sr)
    envelope = resample(envelope, n_target)

    # Z-score
    envelope = (envelope - envelope.mean()) / (envelope.std() + 1e-8)
    return envelope


# ---------------------------------------------------------------------------
# EEG data loading
# ---------------------------------------------------------------------------

def load_all_eeg() -> tuple[dict[tuple[str, int], np.ndarray], dict[int, int], int]:
    """Load all continuous EEG directly from imputed .mat files (no per-epoch normalization).

    Returns:
        eeg_data: dict mapping (subject_id, song_id) -> (T, n_channels) array
        song_lengths: dict mapping song_id -> T
        n_channels: number of EEG channels
    """
    eeg_data = {}
    song_lengths = {}
    n_channels = None

    for song in SONGS:
        mat_path = NMED_DATA_DIR / f"song{song.file_number}_Imputed.mat"
        mat = sio.loadmat(str(mat_path))
        data_key = f"data{song.file_number}"
        subs_key = f"subs{song.file_number}"

        eeg_array = mat[data_key]  # (n_channels, T, n_subjects)
        subject_ids = [mat[subs_key][0, i][0] for i in range(mat[subs_key].shape[1])]

        n_ch, T, n_subj = eeg_array.shape
        if n_channels is None:
            n_channels = n_ch

        song_lengths[song.id] = T

        for s_idx, subj_id in enumerate(subject_ids):
            # (n_channels, T) -> (T, n_channels) — no normalization!
            eeg_data[(subj_id, song.id)] = eeg_array[:, :, s_idx].T.astype(np.float64)

    return eeg_data, song_lengths, n_channels


# ---------------------------------------------------------------------------
# GPU-accelerated lag matrix + ridge regression
# ---------------------------------------------------------------------------

def build_lag_matrix_2d_torch(x: torch.Tensor, lags: torch.Tensor) -> torch.Tensor:
    """Build lagged design matrix on GPU. x: (T, C), lags: (L,) -> (T_valid, C*L)"""
    T, C = x.shape
    L = len(lags)
    min_lag = int(lags.min().item())
    max_lag = int(lags.max().item())
    t_start = max(0, -min_lag)
    t_end = T - max(0, max_lag)
    T_valid = t_end - t_start

    # Gather all lagged versions at once
    cols = []
    for lag in lags:
        lag = int(lag.item())
        cols.append(x[t_start + lag : t_end + lag, :])
    # Stack: (T_valid, L, C) -> reshape to (T_valid, C*L) with lag as inner dim
    stacked = torch.stack(cols, dim=1)  # (T_valid, L, C)
    return stacked.reshape(T_valid, C * L), t_start, t_end


def build_lag_matrix_1d_torch(x: torch.Tensor, lags: torch.Tensor) -> torch.Tensor:
    """Build lagged design matrix on GPU for 1D signal. x: (T,), lags: (L,) -> (T_valid, L)"""
    T = len(x)
    max_lag = int(lags.max().item())
    min_lag = int(lags.min().item())
    t_start = max(0, max_lag)
    t_end = T - max(0, -min_lag)

    cols = []
    for lag in lags:
        lag = int(lag.item())
        cols.append(x[t_start - lag : t_end - lag])
    return torch.stack(cols, dim=1), t_start, t_end


def ridge_solve(XtX: torch.Tensor, Xty: torch.Tensor, alpha: float) -> torch.Tensor:
    """Solve ridge: w = (X'X + αI)^{-1} X'y on GPU."""
    n = XtX.shape[0]
    reg = alpha * torch.eye(n, device=XtX.device, dtype=XtX.dtype)
    return torch.linalg.solve(XtX + reg, Xty)


def pearson_r_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Pearson correlation on GPU."""
    y_true = y_true - y_true.mean()
    y_pred = y_pred - y_pred.mean()
    num = (y_true * y_pred).sum()
    den = torch.sqrt((y_true ** 2).sum() * (y_pred ** 2).sum())
    if den < 1e-10:
        return 0.0
    return float((num / den).item())


# ---------------------------------------------------------------------------
# Backward TRF (GPU)
# ---------------------------------------------------------------------------

def precompute_backward_data(
    subject_id: str,
    eeg_data: dict,
    envelopes: dict[int, np.ndarray],
    lags: torch.Tensor,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Precompute lag matrices and targets for backward TRF on GPU."""
    song_ids = sorted([sid for (subj, sid) in eeg_data if subj == subject_id])
    song_X = {}
    song_y = {}

    min_lag = int(lags.min().item())
    max_lag = int(lags.max().item())

    for song_id in song_ids:
        eeg = eeg_data[(subject_id, song_id)]
        env = envelopes[song_id]
        min_len = min(len(eeg), len(env))

        eeg_t = torch.from_numpy(eeg[:min_len].astype(np.float32)).to(DEVICE)
        env_t = torch.from_numpy(env[:min_len].astype(np.float32)).to(DEVICE)

        X, t_start, t_end = build_lag_matrix_2d_torch(eeg_t, lags)
        y = env_t[t_start:t_end]

        song_X[song_id] = X
        song_y[song_id] = y

    return song_X, song_y


def run_backward_trf_gpu(
    song_X: dict[int, torch.Tensor],
    song_y: dict[int, torch.Tensor],
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run backward TRF leave-one-song-out CV on GPU.

    Returns per_song_r (n_songs,), weights (n_features,)
    """
    song_ids = sorted(song_X.keys())
    per_song_r = np.zeros(len(song_ids))

    # Precompute X'X and X'y per song for efficient fold assembly
    song_XtX = {}
    song_Xty = {}
    for sid in song_ids:
        song_XtX[sid] = song_X[sid].T @ song_X[sid]
        song_Xty[sid] = song_X[sid].T @ song_y[sid]

    total_XtX = sum(song_XtX[s] for s in song_ids)
    total_Xty = sum(song_Xty[s] for s in song_ids)

    # Leave-one-song-out: subtract test song from totals
    for fold_idx, test_song in enumerate(song_ids):
        train_XtX = total_XtX - song_XtX[test_song]
        train_Xty = total_Xty - song_Xty[test_song]
        w = ridge_solve(train_XtX, train_Xty, alpha)
        y_pred = song_X[test_song] @ w
        per_song_r[fold_idx] = pearson_r_torch(song_y[test_song], y_pred)

    # All-data weights
    weights = ridge_solve(total_XtX, total_Xty, alpha)
    return per_song_r, weights.cpu().numpy()


def run_backward_trf_shifted_gpu(
    song_X: dict[int, torch.Tensor],
    envelopes: dict[int, np.ndarray],
    eeg_data_subj: dict[int, np.ndarray],
    lags: torch.Tensor,
    alpha: float,
    shifts: dict[int, int],
) -> float:
    """Run backward TRF with shifted envelopes. Returns mean r across songs."""
    song_ids = sorted(song_X.keys())
    min_lag = int(lags.min().item())
    max_lag = int(lags.max().item())

    # Build shifted targets (only y changes, X stays the same)
    shifted_y = {}
    for sid in song_ids:
        env = envelopes[sid]
        shifted_env = np.roll(env, shifts[sid])
        eeg_len = len(eeg_data_subj[sid])
        min_len = min(eeg_len, len(shifted_env))
        t_start = max(0, -min_lag)
        t_end = min_len - max(0, max_lag)
        shifted_y[sid] = torch.from_numpy(
            shifted_env[t_start:t_end].astype(np.float32)
        ).to(DEVICE)

    # Precompute X'y for shifted targets
    song_XtX = {}
    song_Xty = {}
    for sid in song_ids:
        X = song_X[sid]
        y = shifted_y[sid]
        # Trim to matching length (X and shifted_y might differ slightly)
        n = min(len(X), len(y))
        X_trimmed = X[:n]
        y_trimmed = y[:n]
        song_XtX[sid] = X_trimmed.T @ X_trimmed
        song_Xty[sid] = X_trimmed.T @ y_trimmed

    total_XtX = sum(song_XtX[s] for s in song_ids)
    total_Xty = sum(song_Xty[s] for s in song_ids)

    per_song_r = []
    for test_song in song_ids:
        train_XtX = total_XtX - song_XtX[test_song]
        train_Xty = total_Xty - song_Xty[test_song]
        w = ridge_solve(train_XtX, train_Xty, alpha)
        n = min(len(song_X[test_song]), len(shifted_y[test_song]))
        y_pred = song_X[test_song][:n] @ w
        per_song_r.append(pearson_r_torch(shifted_y[test_song][:n], y_pred))

    return float(np.mean(per_song_r))


# ---------------------------------------------------------------------------
# Forward TRF (GPU) — batched across channels
# ---------------------------------------------------------------------------

def run_forward_trf_gpu(
    subject_id: str,
    eeg_data: dict,
    envelopes: dict[int, np.ndarray],
    lags: torch.Tensor,
    alpha: float,
    n_channels: int = 125,
) -> tuple[np.ndarray, np.ndarray]:
    """Run forward TRF on GPU, predicting all channels at once."""
    song_ids = sorted([sid for (subj, sid) in eeg_data if subj == subject_id])

    song_X = {}
    song_Y = {}
    song_XtX = {}
    song_XtY = {}

    max_lag = int(lags.max().item())
    min_lag = int(lags.min().item())

    for sid in song_ids:
        eeg = eeg_data[(subject_id, sid)]
        env = envelopes[sid]
        min_len = min(len(eeg), len(env))
        env_t = torch.from_numpy(env[:min_len].astype(np.float32)).to(DEVICE)
        eeg_t = torch.from_numpy(eeg[:min_len].astype(np.float32)).to(DEVICE)

        X, t_start, t_end = build_lag_matrix_1d_torch(env_t, lags)
        Y = eeg_t[t_start:t_end, :]

        song_X[sid] = X
        song_Y[sid] = Y
        song_XtX[sid] = X.T @ X
        song_XtY[sid] = X.T @ Y  # (n_lags, 32) — solves all channels at once

    total_XtX = sum(song_XtX[s] for s in song_ids)
    total_XtY = sum(song_XtY[s] for s in song_ids)

    per_song_r = np.zeros((len(song_ids), n_channels))

    for fold_idx, test_song in enumerate(song_ids):
        train_XtX = total_XtX - song_XtX[test_song]
        train_XtY = total_XtY - song_XtY[test_song]

        # Solve all 32 channels at once: W = (X'X + αI)^{-1} X'Y  shape (n_lags, 32)
        n = train_XtX.shape[0]
        reg = alpha * torch.eye(n, device=DEVICE)
        W = torch.linalg.solve(train_XtX + reg, train_XtY)

        Y_pred = song_X[test_song] @ W  # (T_valid, 32)
        Y_true = song_Y[test_song]

        for ch in range(n_channels):
            per_song_r[fold_idx, ch] = pearson_r_torch(Y_true[:, ch], Y_pred[:, ch])

    # All-data weights
    n = total_XtX.shape[0]
    reg = alpha * torch.eye(n, device=DEVICE)
    W_all = torch.linalg.solve(total_XtX + reg, total_XtY)  # (n_lags, 32)
    weights = W_all.T.cpu().numpy()  # (32, n_lags)

    return per_song_r, weights


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TRF analysis for NMED EEG")
    parser.add_argument("--alpha", type=float, default=1e4)
    parser.add_argument("--n-perms", type=int, default=200)
    parser.add_argument("--pca-components", type=int, default=0,
                        help="PCA dimensionality reduction (0=no PCA, e.g. 64)")
    parser.add_argument("--skip-permutation", action="store_true")
    parser.add_argument("--skip-forward", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")

    lags_backward = np.arange(-12, 63)  # 75 lags, -96ms to 496ms
    lags_forward = np.arange(0, 63)     # 63 lags, 0 to 496ms
    lags_backward_t = torch.tensor(lags_backward, device=DEVICE)
    lags_forward_t = torch.tensor(lags_forward, device=DEVICE)

    # --- Extract audio envelopes ---
    t0 = time.time()
    print("Extracting audio envelopes...")
    envelopes = {}
    for song in SONGS:
        audio_path = NMED_AUDIO_DIR / f"song_{song.id:02d}.flac"
        env = extract_envelope(audio_path)
        envelopes[song.id] = env
        print(f"  Song {song.id} ({song.title}): {len(env)} samples ({len(env)/SFREQ:.1f}s)")
    print(f"  [{time.time()-t0:.1f}s]")

    # --- Load EEG ---
    t0 = time.time()
    print("\nLoading continuous EEG from imputed .mat files (no per-epoch normalization)...")
    eeg_data, song_lengths, n_channels = load_all_eeg()
    print(f"  Loaded {len(eeg_data)} subject-song pairs, {n_channels} channels [{time.time()-t0:.1f}s]")

    # Alignment check
    for song_id in sorted(song_lengths.keys()):
        eeg_len = song_lengths[song_id]
        env_len = len(envelopes[song_id])
        diff = abs(eeg_len - env_len)
        status = "OK" if diff < SFREQ else "MISMATCH"
        print(f"  Song {song_id}: EEG={eeg_len} env={env_len} diff={diff} [{status}]")

    # --- Optional PCA dimensionality reduction ---
    if args.pca_components > 0:
        t0_pca = time.time()
        n_comp = args.pca_components
        print(f"\nApplying PCA: {n_channels} -> {n_comp} components (per subject)...")
        from sklearn.decomposition import PCA
        subjects_all = sorted(set(subj for subj, _ in eeg_data))
        for subj in subjects_all:
            # Concatenate all songs for this subject to fit PCA
            subj_songs = sorted([sid for (s, sid) in eeg_data if s == subj])
            all_eeg = np.concatenate([eeg_data[(subj, sid)] for sid in subj_songs], axis=0)
            pca = PCA(n_components=n_comp)
            pca.fit(all_eeg)
            var_explained = pca.explained_variance_ratio_.sum()
            # Transform each song
            for sid in subj_songs:
                eeg_data[(subj, sid)] = pca.transform(eeg_data[(subj, sid)])
            print(f"  {subj}: {var_explained:.1%} variance explained")
        n_channels = n_comp
        print(f"  [{time.time()-t0_pca:.1f}s]")

    subjects = sorted(set(subj for subj, _ in eeg_data))
    n_subjects = len(subjects)
    song_ids = sorted(set(sid for _, sid in eeg_data))
    n_songs = len(song_ids)
    print(f"\n{n_subjects} subjects, {n_songs} songs")

    # --- Backward TRF ---
    print(f"\n{'='*60}")
    print(f"  BACKWARD TRF (Stimulus Reconstruction) — GPU")
    print(f"  Lags: {lags_backward[0]*1000/SFREQ:.0f}ms to {lags_backward[-1]*1000/SFREQ:.0f}ms")
    print(f"  Alpha: {args.alpha:.0e}")
    print(f"{'='*60}")

    t0 = time.time()
    backward_r = np.zeros((n_subjects, n_songs))
    backward_weights = []

    for s_idx, subject_id in enumerate(subjects):
        song_X, song_y = precompute_backward_data(
            subject_id, eeg_data, envelopes, lags_backward_t
        )

        per_song_r, weights = run_backward_trf_gpu(song_X, song_y, args.alpha)
        backward_r[s_idx] = per_song_r
        backward_weights.append(weights)

        # Free GPU memory for this subject's lag matrices
        del song_X, song_y
        torch.cuda.empty_cache()

        mean_r = per_song_r.mean()
        print(f"  {subject_id}: mean r = {mean_r:.4f}  "
              f"[{', '.join(f'{r:.3f}' for r in per_song_r)}]")

    backward_weights = np.stack(backward_weights)
    grand_mean_r = backward_r.mean()
    grand_se = backward_r.mean(axis=1).std() / np.sqrt(n_subjects)
    print(f"\n  Grand average r = {grand_mean_r:.4f} +/- {grand_se:.4f}  [{time.time()-t0:.1f}s]")

    # --- Forward TRF ---
    forward_r = None
    forward_weights = None

    if not args.skip_forward:
        print(f"\n{'='*60}")
        print(f"  FORWARD TRF (Encoding Model) — GPU, batched {n_channels}ch")
        print(f"{'='*60}")

        t0 = time.time()
        forward_r = np.zeros((n_subjects, n_songs, n_channels))
        forward_weights_list = []

        for s_idx, subject_id in enumerate(subjects):
            per_song_r, weights = run_forward_trf_gpu(
                subject_id, eeg_data, envelopes, lags_forward_t, args.alpha,
                n_channels=n_channels,
            )
            forward_r[s_idx] = per_song_r
            forward_weights_list.append(weights)

            mean_r_per_ch = per_song_r.mean(axis=0)
            top_ch_idx = np.argsort(mean_r_per_ch)[-3:][::-1]
            top_str = ", ".join(
                f"ch{c}={mean_r_per_ch[c]:.4f}" for c in top_ch_idx
            )
            print(f"  {subject_id}: best channels: {top_str}")

        forward_weights = np.stack(forward_weights_list)
        mean_per_ch = forward_r.mean(axis=(0, 1))
        top_5 = np.argsort(mean_per_ch)[-5:][::-1]
        print(f"\n  Top 5 channels (grand avg):")
        for c in top_5:
            print(f"    ch{c}: r = {mean_per_ch[c]:.4f}")
        print(f"  [{time.time()-t0:.1f}s]")

    # --- Permutation test ---
    null_r = None
    if not args.skip_permutation:
        print(f"\n{'='*60}")
        print(f"  PERMUTATION TEST ({args.n_perms} permutations) — GPU")
        print(f"{'='*60}")

        t0 = time.time()
        rng = np.random.default_rng(42)
        null_r = np.zeros((n_subjects, args.n_perms))
        min_shift = 30 * SFREQ

        for s_idx, subject_id in enumerate(subjects):
            # Recompute lag matrices for this subject (freed after backward TRF)
            song_X, song_y = precompute_backward_data(
                subject_id, eeg_data, envelopes, lags_backward_t
            )
            eeg_data_subj = {
                sid: eeg_data[(subject_id, sid)]
                for sid in sorted(song_X.keys())
            }

            for perm in range(args.n_perms):
                shifts = {}
                for sid in song_X.keys():
                    env_len = len(envelopes[sid])
                    max_shift = env_len - min_shift
                    shifts[sid] = int(rng.integers(min_shift, max(min_shift + 1, max_shift)))

                null_r[s_idx, perm] = run_backward_trf_shifted_gpu(
                    song_X, envelopes, eeg_data_subj, lags_backward_t, args.alpha, shifts
                )

            # Free GPU memory
            del song_X, song_y
            torch.cuda.empty_cache()

            actual_r = backward_r[s_idx].mean()
            p_val = (1 + np.sum(null_r[s_idx] >= actual_r)) / (1 + args.n_perms)
            sig = "*" if p_val < 0.05 else " "
            print(f"  {subject_id}: actual={actual_r:.4f}, "
                  f"null={null_r[s_idx].mean():.4f}+/-{null_r[s_idx].std():.4f}, "
                  f"p={p_val:.4f} {sig}")

        print(f"  [{time.time()-t0:.1f}s]")

    # --- Save results ---
    save_dict = {
        "backward_r": backward_r,
        "backward_weights": backward_weights,
        "lags_backward": lags_backward,
        "lags_forward": lags_forward,
        "lags_backward_ms": lags_backward * (1000.0 / SFREQ),
        "lags_forward_ms": lags_forward * (1000.0 / SFREQ),
        "subject_ids": np.array(subjects),
        "song_ids": np.array(song_ids),
        "channel_names": np.array([f"ch{i}" for i in range(n_channels)]),
        "n_channels": np.array(n_channels),
        "alpha": np.array(args.alpha),
    }
    if forward_r is not None:
        save_dict["forward_r"] = forward_r
        save_dict["forward_weights"] = forward_weights
    if null_r is not None:
        save_dict["null_r"] = null_r

    results_path = RESULTS_DIR / "results.npz"
    np.savez(str(results_path), **save_dict)
    print(f"\nResults saved to {results_path}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Backward TRF (stimulus reconstruction):")
    print(f"    Grand mean r = {grand_mean_r:.4f} +/- {grand_se:.4f} (SE)")
    print(f"    Range: {backward_r.mean(axis=1).min():.4f} to {backward_r.mean(axis=1).max():.4f}")

    if null_r is not None:
        overall_actual = backward_r.mean(axis=1)
        n_significant = sum(
            (1 + np.sum(null_r[i] >= overall_actual[i])) / (1 + args.n_perms) < 0.05
            for i in range(n_subjects)
        )
        print(f"    Significant subjects (p<0.05): {n_significant}/{n_subjects}")
        print(f"    Null mean r = {null_r.mean():.4f}")

    if forward_r is not None:
        print(f"\n  Forward TRF (encoding model):")
        print(f"    Grand mean r (best channel per subject): "
              f"{forward_r.mean(axis=1).max(axis=1).mean():.4f}")

    print(f"\n  Literature reference: music envelope tracking typically r ~ 0.05-0.15")
    if grand_mean_r > 0.02:
        print(f"  --> Signal DETECTED. EEG contains decodable auditory information.")
    else:
        print(f"  --> Signal NOT detected. EEG-audio relationship below detection threshold.")


if __name__ == "__main__":
    main()
