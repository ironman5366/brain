"""
Evaluate an EEG → Audio Embedding checkpoint on the validation set.

Reports: MSE, per-channel cosine similarity, and per-song breakdown.

Usage:
    uv run python eval_audio_embed.py checkpoints/nmed-audio-embed-v1/epoch_15
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from safetensors.torch import load_file
import polars as pl
from tqdm import tqdm
import numpy as np

from models.audio_embed import EEGAudioEmbed
from data.dataset import SparseAudioEmbedDataset
from data.nmed.songs import NMED_OUTPUT_DIR, SONG_BY_ID


VAL_EEG_PATH = NMED_OUTPUT_DIR / "nmed-val.safetensors"
VAL_ENCODEC_PATH = NMED_OUTPUT_DIR / "nmed-val-encodec.safetensors"
VAL_META_PATH = NMED_OUTPUT_DIR / "nmed-val-metadata.parquet"


def cosine_sim_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between flattened pred/target per sample."""
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    return F.cosine_similarity(pred_flat, target_flat, dim=1)


def evaluate(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = EEGAudioEmbed.from_pretrained(checkpoint_path)
    model = model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  {param_count:,} parameters")

    # Load val dataset
    print(f"Loading val data...")
    dataset = SparseAudioEmbedDataset(VAL_EEG_PATH, str(VAL_ENCODEC_PATH))
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    metadata = pl.read_parquet(VAL_META_PATH)

    # Collect predictions
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for eeg, audio_embeds in tqdm(dataloader, desc="Evaluating"):
            eeg = eeg.to(device)
            audio_embeds = audio_embeds.to(device)
            pred = model(eeg)
            all_preds.append(pred.cpu())
            all_targets.append(audio_embeds.cpu())

    all_preds = torch.cat(all_preds, dim=0)      # (N, 128, 75)
    all_targets = torch.cat(all_targets, dim=0)   # (N, 128, 75)

    # Overall metrics
    mse = F.mse_loss(all_preds, all_targets).item()
    cos_sims = cosine_sim_per_sample(all_preds, all_targets)
    mean_cos = cos_sims.mean().item()
    median_cos = cos_sims.median().item()

    # Correlation across the full embedding (flatten all samples)
    pred_flat = all_preds.numpy().ravel()
    target_flat = all_targets.numpy().ravel()
    pearson_r = np.corrcoef(pred_flat, target_flat)[0, 1]

    print(f"\n{'='*50}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Val samples: {len(all_preds)}")
    print(f"{'='*50}")
    print(f"  MSE loss:          {mse:.4f}")
    print(f"  Cosine sim (mean): {mean_cos:.4f}")
    print(f"  Cosine sim (med):  {median_cos:.4f}")
    print(f"  Pearson r:         {pearson_r:.4f}")

    # Per-song breakdown
    song_ids = metadata["song_id"].to_list()
    unique_songs = sorted(set(song_ids))

    print(f"\n  Per-song breakdown:")
    print(f"  {'Song':<40s} {'N':>5s} {'MSE':>8s} {'CosSim':>8s}")
    print(f"  {'-'*40} {'-'*5} {'-'*8} {'-'*8}")

    for sid in unique_songs:
        mask = torch.tensor([s == sid for s in song_ids])
        song_preds = all_preds[mask]
        song_targets = all_targets[mask]
        song_mse = F.mse_loss(song_preds, song_targets).item()
        song_cos = cosine_sim_per_sample(song_preds, song_targets).mean().item()
        song_name = SONG_BY_ID[sid].title
        print(f"  {song_name:<40s} {mask.sum().item():>5d} {song_mse:>8.4f} {song_cos:>8.4f}")

    # Per-frame MSE (how well does it predict each of the 75 time frames?)
    frame_mse = (all_preds - all_targets).pow(2).mean(dim=(0, 1))  # (75,)
    print(f"\n  Frame MSE range: {frame_mse.min():.4f} - {frame_mse.max():.4f}")
    print(f"  Frame MSE std:   {frame_mse.std():.4f}")

    # Baseline: predict the mean target embedding
    mean_target = all_targets.mean(dim=0, keepdim=True)  # (1, 128, 75)
    baseline_mse = F.mse_loss(mean_target.expand_as(all_targets), all_targets).item()
    baseline_cos = cosine_sim_per_sample(
        mean_target.expand_as(all_targets), all_targets
    ).mean().item()

    print(f"\n  Baseline (predict mean):")
    print(f"    MSE:  {baseline_mse:.4f}")
    print(f"    CosSim: {baseline_cos:.4f}")
    print(f"    MSE improvement: {(1 - mse/baseline_mse)*100:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python eval_audio_embed.py <checkpoint_path>")
        sys.exit(1)
    evaluate(sys.argv[1])
