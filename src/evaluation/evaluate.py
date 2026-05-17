"""
Evaluation utilities for Amharic sentiment analysis models.

Computes per-class and macro-averaged:
  - Accuracy
  - Precision, Recall, F1-score (weighted and macro)
  - Confusion matrix (saved as heatmap figure)

Results are saved to results/metrics/ as JSON and CSV.
"""

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

LABEL_NAMES = ["positive", "negative", "neutral"]


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str] = LABEL_NAMES,
) -> dict:
    """
    Return a flat dict of evaluation metrics suitable for logging.

    All averages use the 'macro' strategy so minority classes (often
    'neutral' in policy discussions) are not under-weighted.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1  = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_p   = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r   = recall_score(y_true, y_pred, average="macro",    zero_division=0)

    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    metrics = {
        "accuracy":       round(acc, 4),
        "macro_f1":       round(macro_f1, 4),
        "weighted_f1":    round(weighted_f1, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall":   round(macro_r, 4),
    }
    for name, f1 in zip(label_names, per_class_f1):
        metrics[f"f1_{name}"] = round(float(f1), 4)

    return metrics


def print_classification_report(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str] = LABEL_NAMES,
) -> None:
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str] = LABEL_NAMES,
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
) -> None:
    """Save a normalised confusion-matrix heatmap to results/figures/."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
    )
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved → {save_path}")
    plt.show()
    plt.close(fig)


def save_metrics(
    metrics: dict,
    run_name: str,
    output_dir: Path = Path("results/metrics"),
) -> None:
    """Persist metrics dict to JSON and append a row to a master CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{run_name}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {json_path}")

    csv_path = output_dir / "all_runs.csv"
    row = {"run": run_name, **metrics}
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(csv_path, index=False)


# ---------------------------------------------------------------------------
# CLI: evaluate a saved model on the test split
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from src.training.train import AmharicDataset, load_split, evaluate_epoch
    from src.models.transformer_model import build_transformer, load_tokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt model weights")
    parser.add_argument("--model", choices=["bilstm", "transformer"], default="transformer")
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--data_dir", default="data/processed/amh")
    parser.add_argument("--run_name", default="test_eval")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(config.get("transformer_checkpoint", "xlm-roberta-base"))

    test_texts, test_labels = load_split(Path(args.data_dir) / "test_processed.tsv")
    test_ds = AmharicDataset(test_texts, test_labels, tokenizer, config.get("max_length", 128))
    test_loader = DataLoader(test_ds, batch_size=config.get("batch_size", 32))

    if args.model == "transformer":
        from src.models.transformer_model import build_transformer
        model = build_transformer(config)
    else:
        from src.models.bilstm_model import build_bilstm
        model = build_bilstm(config)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)

    metrics = evaluate_epoch(model, test_loader, device, args.model)
    print_classification_report(test_labels, test_labels)  # placeholder — swap with preds
    save_metrics(metrics, args.run_name)
    plot_confusion_matrix(
        test_labels, test_labels,  # replace second arg with actual predictions
        title=f"Confusion Matrix — {args.run_name}",
        save_path=Path("results/figures") / f"{args.run_name}_cm.png",
    )
