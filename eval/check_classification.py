# Builtin imports
from pathlib import Path
import argparse
import tomllib

# Internal imports
from data.deap import DEAPClassificationDataset, DEAP_OUT_DIR
from data.dataset import SparseClassificationDataset
from models.ccnn import CCNN1DClassifier

# External imports
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

BATCH_SIZE = 256
DEVICE = "cuda"


def main():
    parser = argparse.ArgumentParser(description="Evaluate classification model")
    parser.add_argument("config", type=str, help="Path to config toml file")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint (default: checkpoints/<config_name>/final)")
    args = parser.parse_args()

    # Parse config
    config_path = Path(args.config)
    config = tomllib.loads(config_path.read_text())

    dataset_type = config["dataset"]
    data_path = config["data_path"]
    class_col = config.get("class_col")

    # Default checkpoint path based on config name
    config_name = config.get("name", config_path.stem)
    checkpoint_path = args.checkpoint or f"checkpoints/{config_name}/final"

    with torch.inference_mode():
        print(f"Loading checkpoint from {checkpoint_path}...")
        model = CCNN1DClassifier.from_pretrained(checkpoint_path).to(DEVICE)
        model.eval()

        if dataset_type == "deap_classification":
            print(f"Loading DEAP validation dataset (class: {class_col})...")
            ds = DEAPClassificationDataset(DEAP_OUT_DIR, split="val", class_col=class_col)
        elif dataset_type == "sparse_classification":
            # Config points to train, swap to val for evaluation
            val_path = data_path.replace("-train.safetensors", "-val.safetensors")
            print(f"Loading sparse dataset from {val_path} (class: {class_col})...")
            ds = SparseClassificationDataset(Path(val_path), class_col=class_col)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        dl = DataLoader(ds, num_workers=8, shuffle=False, batch_size=BATCH_SIZE)

        all_preds = []
        all_targets = []
        all_probs = []

        for batch in tqdm(dl):
            samples, labels = batch
            samples = samples.to(DEVICE)

            logits = model(samples)
            probs = torch.softmax(logits, dim=-1)
            predictions = logits.argmax(dim=-1)

            all_preds.append(predictions.cpu())
            all_targets.append(labels)
            all_probs.append(probs.cpu())

        # Concatenate all
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        probs = torch.cat(all_probs)

        # Compute metrics
        correct = (preds == targets).sum().item()
        total = len(targets)
        accuracy = correct / total

        # Per-class metrics
        num_classes = probs.shape[1]
        class_names = (
            ["Low", "High"]
            if num_classes == 2
            else [str(i) for i in range(num_classes)]
        )

        # Class distribution
        target_counts = [(targets == i).sum().item() for i in range(num_classes)]
        pred_counts = [(preds == i).sum().item() for i in range(num_classes)]

        # Per-class precision, recall, F1
        per_class_precision = []
        per_class_recall = []
        per_class_f1 = []
        for c in range(num_classes):
            tp = ((preds == c) & (targets == c)).sum().item()
            fp = ((preds == c) & (targets != c)).sum().item()
            fn = ((preds != c) & (targets == c)).sum().item()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            per_class_precision.append(prec)
            per_class_recall.append(rec)
            per_class_f1.append(f1)

        # Macro averages
        macro_precision = sum(per_class_precision) / num_classes
        macro_recall = sum(per_class_recall) / num_classes
        macro_f1 = sum(per_class_f1) / num_classes

        # Print results
        print(f"\n{'=' * 50}")
        print(f"Results on {total:,} samples ({class_col})")
        print(f"{'=' * 50}")

        print(f"\n--- Overall Metrics ---")
        print(f"  Accuracy: {accuracy * 100:.2f}% ({correct:,}/{total:,})")
        print(f"  Macro Precision: {macro_precision * 100:.2f}%")
        print(f"  Macro Recall:    {macro_recall * 100:.2f}%")
        print(f"  Macro F1 Score:  {macro_f1 * 100:.2f}%")

        # Confusion matrix for binary, per-class stats for multi-class
        if num_classes == 2:
            tp = ((preds == 1) & (targets == 1)).sum().item()
            tn = ((preds == 0) & (targets == 0)).sum().item()
            fp = ((preds == 1) & (targets == 0)).sum().item()
            fn = ((preds == 0) & (targets == 1)).sum().item()
            print(f"\n--- Confusion Matrix ---")
            print(f"                  Predicted")
            print(f"                  Low    High")
            print(f"  Actual Low   {tn:6,} {fp:6,}")
            print(f"  Actual High  {fn:6,} {tp:6,}")

        print(f"\n--- Per-Class Metrics ---")
        print(f"  {'Class':<8} {'Target':>8} {'Pred':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
        for i, name in enumerate(class_names):
            print(
                f"  {name:<8} {target_counts[i]:>8,} {pred_counts[i]:>8,} "
                f"{per_class_precision[i]*100:>7.1f}% {per_class_recall[i]*100:>7.1f}% {per_class_f1[i]*100:>7.1f}%"
            )

        print(f"\n--- Confidence ---")
        correct_mask = preds == targets
        correct_conf = (
            probs[torch.arange(len(probs)), preds][correct_mask].mean().item()
        )
        incorrect_conf = (
            probs[torch.arange(len(probs)), preds][~correct_mask].mean().item()
            if (~correct_mask).sum() > 0
            else 0
        )
        print(f"  Avg confidence (correct):   {correct_conf * 100:.1f}%")
        print(f"  Avg confidence (incorrect): {incorrect_conf * 100:.1f}%")


if __name__ == "__main__":
    main()
