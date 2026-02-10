"""
Evaluate all 32-channel contrastive checkpoints and produce a comparison report.

Evaluates:
  - 3 baselines (nmed32-only, songfam-only, combined) on 1-second val data
  - 5 hierarchical models (w2, w5, w10, w30, full) on continuous-32ch val data

Outputs a summary table comparing retrieval metrics, song classification, and
cosine similarity across all models.

Usage:
    uv run python eval_all_32ch.py
    uv run python eval_all_32ch.py --output reports/32ch_results.md
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from safetensors.torch import load_file
import polars as pl
from tqdm import tqdm
import numpy as np

from models.audio_contrastive import EEGAudioContrastive
from models.audio_contrastive_hierarchical import HierarchicalEEGAudioContrastive
from data.dataset import DenseAudioEmbedDataset, ContinuousAudioEmbedDataset


DATA_ROOT = Path("/kreka/research/willy/side/brain_datasets/nmed-processed")
CONTINUOUS_ROOT = DATA_ROOT / "continuous-32ch"


@dataclass
class EvalResult:
    name: str
    window: str
    n_val: int
    chance_pct: float
    top1: float
    top5: float
    top10: float
    mrr: float
    median_rank: float
    song_acc: float
    n_songs: int
    cos_mean: float
    cos_std: float


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


def song_classification(
    eeg_embeds: torch.Tensor,
    audio_embeds: torch.Tensor,
    song_ids: list,
) -> tuple[float, int]:
    """N-way song classification by nearest centroid. Returns (accuracy, n_songs)."""
    unique_songs = sorted(set(song_ids))
    song_id_tensor = torch.tensor(song_ids)

    # Mean audio embedding per song
    song_centroids = {}
    for sid in unique_songs:
        mask = song_id_tensor == sid
        song_centroids[sid] = audio_embeds[mask].mean(dim=0)
    centroid_matrix = torch.stack([song_centroids[sid] for sid in unique_songs])
    centroid_matrix = F.normalize(centroid_matrix, dim=-1)

    song_logits = eeg_embeds @ centroid_matrix.T
    song_preds = torch.tensor([unique_songs[i] for i in song_logits.argmax(dim=-1)])
    song_correct = (song_preds == song_id_tensor).float()
    return song_correct.mean().item(), len(unique_songs)


def eval_baseline(checkpoint_path: str, name: str, eeg_path: Path, audio_path: Path,
                  meta_path: Path, song_id_col: str = "song_id") -> EvalResult:
    """Evaluate a 1-second baseline model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EEGAudioContrastive.from_pretrained(checkpoint_path)
    model = model.to(device).eval()

    dataset = DenseAudioEmbedDataset(eeg_path, str(audio_path))
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    metadata = pl.read_parquet(meta_path)

    all_eeg_embeds = []
    all_audio_embeds = []

    with torch.no_grad():
        for eeg, audio in tqdm(dataloader, desc=f"Eval {name}"):
            eeg, audio = eeg.to(device), audio.to(device)
            eeg_embed, audio_embed, _ = model(eeg, audio)
            all_eeg_embeds.append(eeg_embed.cpu())
            all_audio_embeds.append(audio_embed.cpu())

    all_eeg_embeds = torch.cat(all_eeg_embeds, dim=0)
    all_audio_embeds = torch.cat(all_audio_embeds, dim=0)
    N = len(all_eeg_embeds)

    # Retrieval
    sim = all_eeg_embeds @ all_audio_embeds.T
    metrics = retrieval_metrics(sim)

    # Song classification
    if song_id_col == "song_id":
        song_ids = metadata[song_id_col].to_list()
    else:
        # For songfam: map song_filename to integer IDs
        filenames = metadata[song_id_col].to_list()
        unique_files = sorted(set(filenames))
        file_to_id = {f: i for i, f in enumerate(unique_files)}
        song_ids = [file_to_id[f] for f in filenames]

    song_acc, n_songs = song_classification(all_eeg_embeds, all_audio_embeds, song_ids)

    # Cosine similarity
    matched_cos = (all_eeg_embeds * all_audio_embeds).sum(dim=-1)

    return EvalResult(
        name=name,
        window="1s",
        n_val=N,
        chance_pct=100.0 / N,
        top1=metrics["top1"],
        top5=metrics["top5"],
        top10=metrics["top10"],
        mrr=metrics["mrr"],
        median_rank=metrics["median_rank"],
        song_acc=song_acc,
        n_songs=n_songs,
        cos_mean=matched_cos.mean().item(),
        cos_std=matched_cos.std().item(),
    )


def eval_baseline_combined(checkpoint_path: str) -> EvalResult:
    """Evaluate the combined baseline on both NMED-32ch and songfam val sets together."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EEGAudioContrastive.from_pretrained(checkpoint_path)
    model = model.to(device).eval()

    # Load both val datasets
    ds1 = DenseAudioEmbedDataset(
        DATA_ROOT / "nmed-32ch-val.safetensors",
        str(DATA_ROOT / "nmed-32ch-val-encodec.safetensors"),
    )
    ds2 = DenseAudioEmbedDataset(
        DATA_ROOT / "songfam-val.safetensors",
        str(DATA_ROOT / "songfam-val-encodec.safetensors"),
    )
    combined = ConcatDataset([ds1, ds2])
    dataloader = DataLoader(combined, batch_size=128, shuffle=False, num_workers=4)

    # Metadata: assign unique song IDs across both datasets
    meta1 = pl.read_parquet(DATA_ROOT / "nmed-32ch-val-metadata.parquet")
    meta2 = pl.read_parquet(DATA_ROOT / "songfam-val-metadata.parquet")

    # NMED song IDs: 1-10
    nmed_song_ids = meta1["song_id"].to_list()
    # Songfam: map filenames to IDs starting at 100
    songfam_files = meta2["song_filename"].to_list()
    unique_songfam = sorted(set(songfam_files))
    file_to_id = {f: 100 + i for i, f in enumerate(unique_songfam)}
    songfam_song_ids = [file_to_id[f] for f in songfam_files]
    all_song_ids = nmed_song_ids + songfam_song_ids

    all_eeg_embeds = []
    all_audio_embeds = []

    with torch.no_grad():
        for eeg, audio in tqdm(dataloader, desc="Eval combined"):
            eeg, audio = eeg.to(device), audio.to(device)
            eeg_embed, audio_embed, _ = model(eeg, audio)
            all_eeg_embeds.append(eeg_embed.cpu())
            all_audio_embeds.append(audio_embed.cpu())

    all_eeg_embeds = torch.cat(all_eeg_embeds, dim=0)
    all_audio_embeds = torch.cat(all_audio_embeds, dim=0)
    N = len(all_eeg_embeds)

    sim = all_eeg_embeds @ all_audio_embeds.T
    metrics = retrieval_metrics(sim)
    song_acc, n_songs = song_classification(all_eeg_embeds, all_audio_embeds, all_song_ids)
    matched_cos = (all_eeg_embeds * all_audio_embeds).sum(dim=-1)

    return EvalResult(
        name="combined",
        window="1s",
        n_val=N,
        chance_pct=100.0 / N,
        top1=metrics["top1"],
        top5=metrics["top5"],
        top10=metrics["top10"],
        mrr=metrics["mrr"],
        median_rank=metrics["median_rank"],
        song_acc=song_acc,
        n_songs=n_songs,
        cos_mean=matched_cos.mean().item(),
        cos_std=matched_cos.std().item(),
    )


def eval_hierarchical(checkpoint_path: str, window_label: str) -> EvalResult:
    """Evaluate a hierarchical contrastive model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_full = window_label == "full"
    data_dir = CONTINUOUS_ROOT / window_label

    model = HierarchicalEEGAudioContrastive.from_pretrained(checkpoint_path)
    model = model.to(device).eval()

    lengths_path = str(data_dir / "nmed-val-lengths.safetensors") if is_full else None
    dataset = ContinuousAudioEmbedDataset(
        data_dir / "nmed-val.safetensors",
        str(data_dir / "nmed-val-encodec.safetensors"),
        lengths_path=lengths_path,
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
    metadata = pl.read_parquet(data_dir / "nmed-val-metadata.parquet")

    all_eeg_embeds = []
    all_audio_embeds = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Eval hier-{window_label}"):
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

    sim = all_eeg_embeds @ all_audio_embeds.T
    metrics = retrieval_metrics(sim)

    song_ids = metadata["song_id"].to_list()
    song_acc, n_songs = song_classification(all_eeg_embeds, all_audio_embeds, song_ids)
    matched_cos = (all_eeg_embeds * all_audio_embeds).sum(dim=-1)

    return EvalResult(
        name=f"hierarchical-{window_label}",
        window=window_label,
        n_val=N,
        chance_pct=100.0 / N,
        top1=metrics["top1"],
        top5=metrics["top5"],
        top10=metrics["top10"],
        mrr=metrics["mrr"],
        median_rank=metrics["median_rank"],
        song_acc=song_acc,
        n_songs=n_songs,
        cos_mean=matched_cos.mean().item(),
        cos_std=matched_cos.std().item(),
    )


def format_report(results: list[EvalResult]) -> str:
    """Format results as a markdown report."""
    lines = []
    lines.append("# 32-Channel Contrastive Model Evaluation Results\n")

    # Summary table
    lines.append("## Retrieval Metrics (EEG → Audio)\n")
    lines.append("| Model | Window | N_val | Chance% | Top-1% | Top-5% | Top-10% | MRR | Med.Rank |")
    lines.append("|-------|--------|------:|--------:|-------:|-------:|--------:|----:|---------:|")
    for r in results:
        lines.append(
            f"| {r.name} | {r.window} | {r.n_val:,} | {r.chance_pct:.3f} | "
            f"{r.top1*100:.2f} | {r.top5*100:.2f} | {r.top10*100:.2f} | "
            f"{r.mrr:.4f} | {r.median_rank:.0f} |"
        )

    lines.append("")
    lines.append("## Song Classification & Cosine Similarity\n")
    lines.append("| Model | Window | Songs | Song Acc% | Chance% | Cos Mean | Cos Std |")
    lines.append("|-------|--------|------:|----------:|--------:|---------:|--------:|")
    for r in results:
        lines.append(
            f"| {r.name} | {r.window} | {r.n_songs} | "
            f"{r.song_acc*100:.2f} | {100/r.n_songs:.1f} | "
            f"{r.cos_mean:.4f} | {r.cos_std:.4f} |"
        )

    # Key comparisons
    lines.append("")
    lines.append("## Key Comparisons\n")

    baselines = [r for r in results if "hierarchical" not in r.name]
    hierarchicals = [r for r in results if "hierarchical" in r.name]

    if baselines:
        best_baseline = max(baselines, key=lambda r: r.song_acc)
        lines.append(f"- **Best baseline**: {best_baseline.name} — "
                     f"Song acc {best_baseline.song_acc*100:.2f}%, "
                     f"Top-1 {best_baseline.top1*100:.2f}%")

    if hierarchicals:
        best_hier = max(hierarchicals, key=lambda r: r.song_acc)
        lines.append(f"- **Best hierarchical**: {best_hier.name} — "
                     f"Song acc {best_hier.song_acc*100:.2f}%, "
                     f"Top-1 {best_hier.top1*100:.2f}%")

    if baselines and hierarchicals:
        # Compare on NMED-only val (song classification is comparable since both use 10 NMED songs)
        nmed_baseline = next((r for r in baselines if r.name == "nmed32-only"), None)
        combined_baseline = next((r for r in baselines if r.name == "combined"), None)
        ref = nmed_baseline or combined_baseline or baselines[0]
        lines.append(f"\n### Song Classification (10-way NMED) Comparison")
        lines.append(f"- Baseline ({ref.name}): {ref.song_acc*100:.2f}%")
        for h in hierarchicals:
            delta = (h.song_acc - ref.song_acc) * 100
            sign = "+" if delta >= 0 else ""
            lines.append(f"- {h.name}: {h.song_acc*100:.2f}% ({sign}{delta:.2f}pp)")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate all 32ch contrastive models")
    parser.add_argument("--output", type=str, default="reports/32ch_eval_results.md",
                        help="Output path for markdown report")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip baseline evaluations")
    parser.add_argument("--skip-hierarchical", action="store_true",
                        help="Skip hierarchical evaluations")
    args = parser.parse_args()

    results = []

    # --- Baselines ---
    if not args.skip_baselines:
        baseline_configs = [
            ("nmed32-only", "checkpoints/contrastive-nmed32-only/final",
             DATA_ROOT / "nmed-32ch-val.safetensors",
             DATA_ROOT / "nmed-32ch-val-encodec.safetensors",
             DATA_ROOT / "nmed-32ch-val-metadata.parquet",
             "song_id"),
            ("songfam-only", "checkpoints/contrastive-songfam-only/final",
             DATA_ROOT / "songfam-val.safetensors",
             DATA_ROOT / "songfam-val-encodec.safetensors",
             DATA_ROOT / "songfam-val-metadata.parquet",
             "song_filename"),
        ]

        for name, ckpt, eeg, audio, meta, sid_col in baseline_configs:
            if Path(ckpt).exists():
                print(f"\n{'='*60}")
                print(f"  Evaluating baseline: {name}")
                print(f"{'='*60}")
                r = eval_baseline(ckpt, name, eeg, audio, meta, sid_col)
                results.append(r)
                print(f"  Top-1: {r.top1*100:.2f}%, Song acc: {r.song_acc*100:.2f}%")
            else:
                print(f"  Skipping {name}: checkpoint not found at {ckpt}")

        # Combined baseline on both val sets
        combined_ckpt = "checkpoints/contrastive-combined/final"
        if Path(combined_ckpt).exists():
            print(f"\n{'='*60}")
            print(f"  Evaluating baseline: combined (on both val sets)")
            print(f"{'='*60}")
            r = eval_baseline_combined(combined_ckpt)
            results.append(r)
            print(f"  Top-1: {r.top1*100:.2f}%, Song acc: {r.song_acc*100:.2f}%")

            # Also eval combined model on NMED-only val for fair comparison with hierarchical
            print(f"\n{'='*60}")
            print(f"  Evaluating baseline: combined (NMED-only val)")
            print(f"{'='*60}")
            r_nmed = eval_baseline(
                combined_ckpt, "combined-nmed-val",
                DATA_ROOT / "nmed-32ch-val.safetensors",
                DATA_ROOT / "nmed-32ch-val-encodec.safetensors",
                DATA_ROOT / "nmed-32ch-val-metadata.parquet",
                "song_id",
            )
            results.append(r_nmed)
            print(f"  Top-1: {r_nmed.top1*100:.2f}%, Song acc: {r_nmed.song_acc*100:.2f}%")

    # --- Hierarchical ---
    if not args.skip_hierarchical:
        window_labels = ["w2", "w5", "w10", "w30", "full"]
        for wl in window_labels:
            ckpt = f"checkpoints/nmed-32ch-hierarchical-contrastive-{wl}/final"
            if Path(ckpt).exists():
                print(f"\n{'='*60}")
                print(f"  Evaluating hierarchical: {wl}")
                print(f"{'='*60}")
                r = eval_hierarchical(ckpt, wl)
                results.append(r)
                print(f"  Top-1: {r.top1*100:.2f}%, Song acc: {r.song_acc*100:.2f}%")
            else:
                print(f"  Skipping hierarchical-{wl}: checkpoint not found at {ckpt}")

    # --- Report ---
    report = format_report(results)
    print(f"\n\n{report}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    main()
