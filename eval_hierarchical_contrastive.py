"""
Evaluate a Hierarchical EEG → Audio Contrastive checkpoint on the validation set.

Reports retrieval metrics (top-k accuracy, MRR), song classification, and cosine similarity.

Usage:
    uv run python eval_hierarchical_contrastive.py <checkpoint_path> <window_label>
    uv run python eval_hierarchical_contrastive.py checkpoints/nmed-hierarchical-contrastive-w5/final w5
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import polars as pl
from tqdm import tqdm

from models.audio_contrastive_hierarchical import HierarchicalEEGAudioContrastive
from data.dataset import ContinuousAudioEmbedDataset
from data.nmed.songs import NMED_OUTPUT_DIR, SONG_BY_ID


def retrieval_metrics(sim_matrix: torch.Tensor) -> dict:
    """Compute retrieval metrics from a similarity matrix."""
    N = sim_matrix.shape[0]
    correct_scores = sim_matrix.diag().unsqueeze(1)
    ranks = (sim_matrix > correct_scores).sum(dim=1)

    top1 = (ranks < 1).float().mean().item()
    top5 = (ranks < 5).float().mean().item()
    top10 = (ranks < 10).float().mean().item()
    mrr = (1.0 / (ranks.float() + 1)).mean().item()
    median_rank = ranks.float().median().item() + 1

    return {
        "top1": top1,
        "top5": top5,
        "top10": top10,
        "mrr": mrr,
        "median_rank": median_rank,
    }


def evaluate(checkpoint_path: str, window_label: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_full = window_label == "full"
    data_dir = NMED_OUTPUT_DIR / "continuous" / window_label

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = HierarchicalEEGAudioContrastive.from_pretrained(checkpoint_path)
    model = model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  {param_count:,} parameters")
    print(f"  logit_scale: {model.log_logit_scale.exp().item():.2f}")

    # Load val dataset
    print("Loading val data...")
    lengths_path = str(data_dir / "nmed-val-lengths.safetensors") if is_full else None
    dataset = ContinuousAudioEmbedDataset(
        data_dir / "nmed-val.safetensors",
        str(data_dir / "nmed-val-encodec.safetensors"),
        lengths_path=lengths_path,
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
    metadata = pl.read_parquet(data_dir / "nmed-val-metadata.parquet")

    # Collect embeddings
    all_eeg_embeds = []
    all_audio_embeds = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Embedding"):
            if len(batch) == 3:
                eeg, audio, lengths = batch
                eeg, audio, lengths = eeg.to(device), audio.to(device), lengths.to(device)
                eeg_embed, audio_embed, _ = model(eeg, audio, lengths)
            else:
                eeg, audio = batch
                eeg, audio = eeg.to(device), audio.to(device)
                eeg_embed, audio_embed, _ = model(eeg, audio)
            all_eeg_embeds.append(eeg_embed.cpu())
            all_audio_embeds.append(audio_embed.cpu())

    all_eeg_embeds = torch.cat(all_eeg_embeds, dim=0)
    all_audio_embeds = torch.cat(all_audio_embeds, dim=0)
    N = len(all_eeg_embeds)

    print(f"\n{'='*60}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Window: {window_label}")
    print(f"  Val sequences: {N}")
    print(f"  Embedding dim: {all_eeg_embeds.shape[1]}")
    print(f"{'='*60}")

    # Full similarity matrix
    print("\nComputing similarity matrix...")
    sim_eeg_to_audio = all_eeg_embeds @ all_audio_embeds.T

    # Overall retrieval metrics
    eeg2audio = retrieval_metrics(sim_eeg_to_audio)
    audio2eeg = retrieval_metrics(sim_eeg_to_audio.T)

    print(f"\n  EEG → Audio Retrieval:")
    print(f"    Top-1:  {eeg2audio['top1']*100:6.2f}%")
    print(f"    Top-5:  {eeg2audio['top5']*100:6.2f}%")
    print(f"    Top-10: {eeg2audio['top10']*100:6.2f}%")
    print(f"    MRR:    {eeg2audio['mrr']:.4f}")
    print(f"    Median rank: {eeg2audio['median_rank']:.0f}")

    print(f"\n  Audio → EEG Retrieval:")
    print(f"    Top-1:  {audio2eeg['top1']*100:6.2f}%")
    print(f"    Top-5:  {audio2eeg['top5']*100:6.2f}%")
    print(f"    Top-10: {audio2eeg['top10']*100:6.2f}%")
    print(f"    MRR:    {audio2eeg['mrr']:.4f}")
    print(f"    Median rank: {audio2eeg['median_rank']:.0f}")

    print(f"\n  Random baseline: top-1 = {1/N*100:.4f}%")

    # Per-song classification
    song_ids = metadata["song_id"].to_list()
    unique_songs = sorted(set(song_ids))
    song_id_tensor = torch.tensor(song_ids)

    # Mean audio embedding per song
    song_centroids = {}
    for sid in unique_songs:
        mask = song_id_tensor == sid
        song_centroids[sid] = all_audio_embeds[mask].mean(dim=0)
    centroid_matrix = torch.stack([song_centroids[sid] for sid in unique_songs])
    centroid_matrix = F.normalize(centroid_matrix, dim=-1)

    song_logits = all_eeg_embeds @ centroid_matrix.T
    song_preds = torch.tensor([unique_songs[i] for i in song_logits.argmax(dim=-1)])
    song_correct = (song_preds == song_id_tensor).float()
    song_acc = song_correct.mean().item()

    print(f"\n  Song classification ({len(unique_songs)}-way):")
    print(f"    Accuracy: {song_acc*100:.2f}% (chance = {100/len(unique_songs):.1f}%)")

    # Per-song breakdown
    print(f"\n  Per-song breakdown:")
    print(f"  {'Song':<40s} {'N':>5s} {'Acc':>8s}")
    print(f"  {'-'*40} {'-'*5} {'-'*8}")
    for sid in unique_songs:
        mask = song_id_tensor == sid
        acc = song_correct[mask].mean().item()
        song_name = SONG_BY_ID[sid].title
        print(f"  {song_name:<40s} {mask.sum().item():>5d} {acc*100:>7.2f}%")

    # Cosine similarity of matched pairs
    matched_cos = (all_eeg_embeds * all_audio_embeds).sum(dim=-1)
    print(f"\n  Cosine similarity (matched pairs):")
    print(f"    Mean:   {matched_cos.mean().item():.4f}")
    print(f"    Median: {matched_cos.median().item():.4f}")
    print(f"    Std:    {matched_cos.std().item():.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: uv run python eval_hierarchical_contrastive.py <checkpoint_path> <window_label>")
        print("  window_label: w2, w5, w10, w30, full")
        sys.exit(1)
    evaluate(sys.argv[1], sys.argv[2])
