"""
Standalone XLM-RoBERTa fine-tuning script for Amharic sentiment analysis.

Data  : data/processed/{train,val,test}.csv
        columns: cleaned_text (str), label (int: 0=positive,1=negative,2=neutral)
Output: results/models/xlmr_best.pt
        results/metrics/xlmr_history.json
        results/metrics/baseline_comparison.csv  (updated)
        results/metrics/final_model_comparison.csv
        results/figures/cm_xlmr.png
        results/figures/xlmr_training_curves.png
"""

import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_CHECKPOINT = "xlm-roberta-base"
DATA_DIR         = Path("data/processed")
RESULTS_DIR      = Path("results")
MODELS_DIR       = RESULTS_DIR / "models"
METRICS_DIR      = RESULTS_DIR / "metrics"
FIGURES_DIR      = RESULTS_DIR / "figures"

BATCH_SIZE       = 16
GRAD_ACCUM       = 2          # effective batch = 32
MAX_LENGTH       = 128
LR               = 2e-5
WARMUP_STEPS     = 100
MAX_EPOCHS       = 10
PATIENCE         = 3
DROPOUT          = 0.1
SEED             = 42

LABEL_NAMES      = ["positive", "negative", "neutral"]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class AmharicTransformerDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = MAX_LENGTH):
        self.labels = torch.tensor(labels, dtype=torch.long)
        enc = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        self.input_ids      = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels":         self.labels[idx],
        }


def load_csv_split(path: Path) -> tuple[list[str], list[int]]:
    df = pd.read_csv(path)
    texts  = df["cleaned_text"].fillna("").tolist()
    labels = df["label"].tolist()
    return texts, labels


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class XLMRobertaSentiment(nn.Module):
    def __init__(self, checkpoint: str = MODEL_CHECKPOINT, num_labels: int = 3, dropout: float = DROPOUT):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(checkpoint)
        hidden_size     = self.encoder.config.hidden_size  # 768 for base
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = self.dropout(outputs.last_hidden_state[:, 0, :])
        logits     = self.classifier(cls_output)
        loss       = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, scheduler, device, grad_accum: int = GRAD_ACCUM) -> float:
    model.train()
    optimizer.zero_grad()
    total_loss   = 0.0
    steps        = 0
    for i, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        out  = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out["loss"] / grad_accum
        loss.backward()
        total_loss += out["loss"].item()
        steps      += 1

        if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return total_loss / steps


def evaluate_epoch(model, loader, device) -> tuple[list[int], list[int], float]:
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            out            = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            preds          = out["logits"].argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())
            total_loss += out["loss"].item()
    return all_labels, all_preds, total_loss / len(loader)


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_training_curves(history: list[dict], save_path: Path):
    epochs     = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]
    val_f1     = [h["val_f1"]     for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_loss, "b-o", label="Train Loss")
    ax1.plot(epochs, val_loss,   "r-o", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("XLM-RoBERTa – Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_f1, "g-o", label="Val F1 (macro)")
    ax2.axhline(y=max(val_f1), color="g", linestyle="--", alpha=0.5, label=f"Best={max(val_f1):.4f}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Macro")
    ax2.set_title("XLM-RoBERTa – Validation F1")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved → {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path: Path):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax,
    )
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("XLM-RoBERTa – Confusion Matrix (Test)")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_seed(SEED)

    for d in [MODELS_DIR, METRICS_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  XLM-RoBERTa Fine-Tuning for Amharic Sentiment")
    print(f"{'='*60}")
    print(f"  Device    : {device}")
    print(f"  Model     : {MODEL_CHECKPOINT}")
    print(f"  Batch size: {BATCH_SIZE}  (grad_accum={GRAD_ACCUM}, effective={BATCH_SIZE*GRAD_ACCUM})")
    print(f"  LR        : {LR}")
    print(f"  Max epochs: {MAX_EPOCHS}  (early-stop patience={PATIENCE})")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Loading data...")
    train_texts, train_labels = load_csv_split(DATA_DIR / "train.csv")
    val_texts,   val_labels   = load_csv_split(DATA_DIR / "val.csv")
    test_texts,  test_labels  = load_csv_split(DATA_DIR / "test.csv")
    print(f"  Train: {len(train_texts)}  |  Val: {len(val_texts)}  |  Test: {len(test_texts)}")

    # ------------------------------------------------------------------
    # 2. Tokenize
    # ------------------------------------------------------------------
    print(f"\nLoading tokenizer ({MODEL_CHECKPOINT})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    print("Tokenizing datasets (this may take a moment)...")
    train_ds = AmharicTransformerDataset(train_texts, train_labels, tokenizer)
    val_ds   = AmharicTransformerDataset(val_texts,   val_labels,   tokenizer)
    test_ds  = AmharicTransformerDataset(test_texts,  test_labels,  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ------------------------------------------------------------------
    # 3. Build model
    # ------------------------------------------------------------------
    print(f"\nLoading {MODEL_CHECKPOINT} + classification head...")
    model = XLMRobertaSentiment(MODEL_CHECKPOINT, num_labels=3, dropout=DROPOUT).to(device)
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params    : {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # ------------------------------------------------------------------
    # 4. Optimizer + linear warmup scheduler
    # ------------------------------------------------------------------
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_update_steps = (len(train_loader) // GRAD_ACCUM) * MAX_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_update_steps,
    )

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    print(f"\n{'─'*60}")
    print("  Epoch  | Train Loss | Val Loss  | Val F1   | Status")
    print(f"{'─'*60}")

    best_val_f1   = 0.0
    best_ckpt     = MODELS_DIR / "xlmr_best.pt"
    patience_cnt  = 0
    history       = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss                      = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_labels_ep, val_preds, val_loss = evaluate_epoch(model, val_loader, device)
        val_f1                          = macro_f1(val_labels_ep, val_preds)

        status = ""
        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            patience_cnt = 0
            torch.save(model.state_dict(), best_ckpt)
            status = "  ← best"
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                                 "val_loss": round(val_loss, 4), "val_f1": round(val_f1, 4)})
                print(f"  {epoch:>5}  | {train_loss:>10.4f} | {val_loss:>9.4f} | {val_f1:>8.4f} | early stop")
                break

        history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                         "val_loss": round(val_loss, 4), "val_f1": round(val_f1, 4)})
        print(f"  {epoch:>5}  | {train_loss:>10.4f} | {val_loss:>9.4f} | {val_f1:>8.4f} |{status}")

    print(f"{'─'*60}")
    print(f"\n  Best val F1 (macro): {best_val_f1:.4f}  →  checkpoint saved")

    # ------------------------------------------------------------------
    # 6. Save training history
    # ------------------------------------------------------------------
    history_path = METRICS_DIR / "xlmr_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Training history saved → {history_path}")

    # ------------------------------------------------------------------
    # 7. Training curves
    # ------------------------------------------------------------------
    plot_training_curves(history, FIGURES_DIR / "xlmr_training_curves.png")

    # ------------------------------------------------------------------
    # 8. Evaluate best model on test set
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  Test Set Evaluation (best checkpoint)")
    print(f"{'='*60}")

    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_labels_true, test_preds, _ = evaluate_epoch(model, test_loader, device)

    acc        = accuracy_score(test_labels_true, test_preds)
    f1_mac     = f1_score(test_labels_true, test_preds, average="macro",    zero_division=0)
    f1_wt      = f1_score(test_labels_true, test_preds, average="weighted", zero_division=0)
    prec       = precision_score(test_labels_true, test_preds, average="macro", zero_division=0)
    rec        = recall_score(test_labels_true, test_preds, average="macro",    zero_division=0)

    print(f"\n  Accuracy   : {acc:.4f}")
    print(f"  Precision  : {prec:.4f}")
    print(f"  Recall     : {rec:.4f}")
    print(f"  F1 macro   : {f1_mac:.4f}")
    print(f"  F1 weighted: {f1_wt:.4f}")

    print("\n  Per-class report:")
    print(classification_report(test_labels_true, test_preds, target_names=LABEL_NAMES, zero_division=0))

    # ------------------------------------------------------------------
    # 9. Confusion matrix
    # ------------------------------------------------------------------
    plot_confusion_matrix(test_labels_true, test_preds, FIGURES_DIR / "cm_xlmr.png")

    # ------------------------------------------------------------------
    # 10. Update baseline_comparison.csv
    # ------------------------------------------------------------------
    xlmr_row = {
        "Model":       "XLM-RoBERTa",
        "Accuracy":    round(acc,   4),
        "Precision":   round(prec,  4),
        "Recall":      round(rec,   4),
        "F1_macro":    round(f1_mac,4),
        "F1_weighted": round(f1_wt, 4),
    }

    baseline_path = METRICS_DIR / "baseline_comparison.csv"
    if baseline_path.exists():
        df_base = pd.read_csv(baseline_path)
        # Remove any pre-existing XLM-RoBERTa row before appending
        df_base = df_base[df_base["Model"] != "XLM-RoBERTa"]
    else:
        df_base = pd.DataFrame(columns=xlmr_row.keys())

    df_base = pd.concat([df_base, pd.DataFrame([xlmr_row])], ignore_index=True)
    df_base = df_base.sort_values("F1_macro", ascending=False).reset_index(drop=True)
    df_base.to_csv(baseline_path, index=False)

    # ------------------------------------------------------------------
    # 11. Save final_model_comparison.csv
    # ------------------------------------------------------------------
    final_path = METRICS_DIR / "final_model_comparison.csv"
    df_base.to_csv(final_path, index=False)

    # ------------------------------------------------------------------
    # 12. Print full comparison table
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  FULL MODEL COMPARISON (sorted by F1 macro)")
    print(f"{'='*60}")
    print(df_base.to_string(index=False))
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 13. Final verdict
    # ------------------------------------------------------------------
    nb_f1    = 0.6109
    bilstm_f1 = 0.5776

    print("  VERDICT")
    print(f"  {'─'*40}")
    beat_nb     = f1_mac > nb_f1
    beat_bilstm = f1_mac > bilstm_f1

    print(f"  XLM-RoBERTa F1_macro : {f1_mac:.4f}")
    print(f"  Naive Bayes  F1_macro : {nb_f1:.4f}  →  XLM-RoBERTa {'BEATS' if beat_nb else 'does NOT beat'} Naive Bayes")
    print(f"  Bi-LSTM      F1_macro : {bilstm_f1:.4f}  →  XLM-RoBERTa {'BEATS' if beat_bilstm else 'does NOT beat'} Bi-LSTM")

    best_model = df_base.iloc[0]["Model"]
    best_f1_overall = df_base.iloc[0]["F1_macro"]
    print(f"\n  BEST MODEL OVERALL: {best_model}  (F1_macro = {best_f1_overall:.4f})")
    print(f"{'='*60}\n")

    print("Done. All results saved to results/")
    return f1_mac


if __name__ == "__main__":
    main()
