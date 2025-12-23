# Builtin imports
from pathlib import Path

# Internal imports
from data.deap import DEAPRegressionDataset, DEAP_OUT_DIR
from models.ccnn import CCNN1DRegressor

# External imports
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

BATCH_SIZE = 256
DEVICE = "cuda"


def main():
    # checkpoint_path = "checkpoints/ccnn_valence/final"
    # label_col = "valence"

    checkpoint_path = "checkpoints/ccnn_arousal/final"
    label_col = "arousal"

    with torch.inference_mode():
        print(f"Loading checkpoint from {checkpoint_path}...")
        model = CCNN1DRegressor.from_pretrained(checkpoint_path).to(DEVICE)
        model.eval()

        print(f"Loading DEAP validation dataset (label: {label_col})...")
        ds = DEAPRegressionDataset(DEAP_OUT_DIR, split="val", label_col=label_col)
        dl = DataLoader(ds, num_workers=8, shuffle=False, batch_size=BATCH_SIZE)

        all_preds = []
        all_targets = []

        for batch in tqdm(dl):
            samples, labels = batch
            samples = samples.to(DEVICE)

            predictions = model(samples)

            all_preds.append(predictions.cpu())
            all_targets.append(labels)

        # Concatenate all predictions and targets
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)

        # Compute metrics
        mse = ((preds - targets) ** 2).mean().item()
        mae = torch.abs(preds - targets).mean().item()
        rmse = mse**0.5

        # Accuracy within X points
        within_05 = (torch.abs(preds - targets) <= 0.5).float().mean().item()
        within_1 = (torch.abs(preds - targets) <= 1.0).float().mean().item()
        within_2 = (torch.abs(preds - targets) <= 2.0).float().mean().item()

        # Binary accuracy (high/low at threshold 5.0)
        pred_high = preds >= 5.0
        target_high = targets >= 5.0
        binary_acc = (pred_high == target_high).float().mean().item()

        # Pearson correlation
        preds_centered = preds - preds.mean()
        targets_centered = targets - targets.mean()
        correlation = (preds_centered * targets_centered).sum() / (
            torch.sqrt((preds_centered**2).sum() * (targets_centered**2).sum()) + 1e-8
        )
        correlation = correlation.item()

        # R² score
        ss_res = ((targets - preds) ** 2).sum()
        ss_tot = ((targets - targets.mean()) ** 2).sum()
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        r2 = r2.item()

        # Print results
        print(f"\n{'=' * 50}")
        print(f"Results on {len(ds):,} samples ({label_col})")
        print(f"{'=' * 50}")

        print(f"\n--- Error Metrics ---")
        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")

        print(f"\n--- Accuracy Metrics ---")
        print(f"  Within 0.5 points: {within_05 * 100:.1f}%")
        print(f"  Within 1.0 points: {within_1 * 100:.1f}%")
        print(f"  Within 2.0 points: {within_2 * 100:.1f}%")
        print(f"  Binary (high/low): {binary_acc * 100:.1f}%")

        print(f"\n--- Correlation Metrics ---")
        print(f"  Pearson r: {correlation:.4f}")
        print(f"  R² score:  {r2:.4f}")

        print(f"\n--- Distribution ---")
        print(f"  Target range: [{targets.min():.2f}, {targets.max():.2f}]")
        print(f"  Pred range:   [{preds.min():.2f}, {preds.max():.2f}]")
        print(f"  Target mean:  {targets.mean():.2f} ± {targets.std():.2f}")
        print(f"  Pred mean:    {preds.mean():.2f} ± {preds.std():.2f}")


if __name__ == "__main__":
    main()
