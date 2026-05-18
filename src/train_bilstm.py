"""Train Bi-LSTM model for Amharic sentiment analysis.

Uses character-level tokenization which solves OOV for Amharic's
morphologically rich Ge'ez script (~500 unique characters vs 36K word forms).
"""

import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "processed")
RESULTS_DIR = os.path.join(ROOT, "results")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

for d in (MODELS_DIR, FIGURES_DIR, METRICS_DIR):
    os.makedirs(d, exist_ok=True)

sys.path.insert(0, ROOT)
from src.models.bilstm_model import AmharicBiLSTM

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAX_LEN = 256          # chars per sample (avg Amharic word ~3 chars)
BATCH_SIZE = 32
EMBED_DIM = 64         # small: only ~600 chars in vocab
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.4
LR = 1e-3
MAX_EPOCHS = 50
EARLY_STOP_PATIENCE = 7
SCHEDULER_PATIENCE = 3
NUM_CLASSES = 3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---------------------------------------------------------------------------
# Character-level vocab builder
# ---------------------------------------------------------------------------

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def build_char_vocab(texts):
    chars = set()
    for text in texts:
        chars.update(list(str(text)))
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for ch in sorted(chars):
        vocab[ch] = len(vocab)
    return vocab


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AmharicCharDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: dict, max_len: int = MAX_LEN):
        self.labels = df["label"].tolist()
        self.vocab = vocab
        self.max_len = max_len
        self.pad_idx = vocab[PAD_TOKEN]
        self.unk_idx = vocab[UNK_TOKEN]
        self.sequences = [self._encode(text) for text in df["cleaned_text"].fillna("").tolist()]

    def _encode(self, text: str):
        ids = [self.vocab.get(ch, self.unk_idx) for ch in list(str(text))]
        ids = ids[: self.max_len]
        ids += [self.pad_idx] * (self.max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, f1, all_preds, all_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load data
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df   = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    # Build character vocab from training text only
    all_train_text = train_df["cleaned_text"].fillna("").tolist()
    vocab = build_char_vocab(all_train_text)
    vocab_size = len(vocab)
    print(f"Character vocab size: {vocab_size}")

    # Save char vocab for reproducibility
    vocab_save_path = os.path.join(DATA_DIR, "char_vocab.json")
    with open(vocab_save_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    train_ds = AmharicCharDataset(train_df, vocab)
    val_ds   = AmharicCharDataset(val_df, vocab)
    test_ds  = AmharicCharDataset(test_df, vocab)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Build model
    model = AmharicBiLSTM(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
        pad_idx=vocab[PAD_TOKEN],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=SCHEDULER_PATIENCE, factor=0.5
    )

    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_val_f1 = 0.0
    best_epoch  = 0
    no_improve  = 0
    best_model_path = os.path.join(MODELS_DIR, "bilstm_best.pt")

    print(f"\n{'Epoch':>5} {'Train Loss':>11} {'Val Loss':>9} {'Val F1':>8} {'LR':>10}")
    print("-" * 55)

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_f1, _, _ = eval_epoch(model, val_loader, criterion)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"{epoch:>5}  {train_loss:>11.4f}  {val_loss:>9.4f}  {val_f1:>8.4f}"
            f"  {current_lr:>10.2e}  ({elapsed:.1f}s)"
        )

        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            no_improve  = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
                break

    print(f"\nBest val F1: {best_val_f1:.4f} at epoch {best_epoch}")
    print(f"Model saved → {best_model_path}")

    # -----------------------------------------------------------------------
    # Evaluate on test set
    # -----------------------------------------------------------------------
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    _, test_f1, test_preds, test_labels = eval_epoch(model, test_loader, criterion)

    acc        = accuracy_score(test_labels, test_preds)
    prec       = precision_score(test_labels, test_preds, average="macro",    zero_division=0)
    rec        = recall_score(test_labels,    test_preds, average="macro",    zero_division=0)
    f1_macro   = f1_score(test_labels,        test_preds, average="macro",    zero_division=0)
    f1_weighted = f1_score(test_labels,       test_preds, average="weighted", zero_division=0)

    print("\n" + "=" * 55)
    print("TEST SET RESULTS")
    print("=" * 55)
    print(f"Accuracy:    {acc:.4f}")
    print(f"Precision:   {prec:.4f}  (macro)")
    print(f"Recall:      {rec:.4f}  (macro)")
    print(f"F1 macro:    {f1_macro:.4f}")
    print(f"F1 weighted: {f1_weighted:.4f}")
    print()
    print(classification_report(
        test_labels, test_preds,
        target_names=["positive", "negative", "neutral"],
        digits=4
    ))

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["positive", "negative", "neutral"],
        yticklabels=["positive", "negative", "neutral"],
    )
    ax.set_title("Bi-LSTM Confusion Matrix (char-level)", fontsize=14)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, "cm_bilstm.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {cm_path}")

    # -----------------------------------------------------------------------
    # Update baseline_comparison.csv
    # -----------------------------------------------------------------------
    comp_path = os.path.join(METRICS_DIR, "baseline_comparison.csv")
    comp_df = pd.read_csv(comp_path)
    comp_df = comp_df[comp_df["Model"] != "Bi-LSTM"]
    new_row = pd.DataFrame([{
        "Model":       "Bi-LSTM",
        "Accuracy":    round(acc,        4),
        "Precision":   round(prec,       4),
        "Recall":      round(rec,        4),
        "F1_macro":    round(f1_macro,   4),
        "F1_weighted": round(f1_weighted, 4),
    }])
    comp_df = pd.concat([comp_df, new_row], ignore_index=True)
    comp_df.to_csv(comp_path, index=False)

    print("\n" + "=" * 55)
    print("UPDATED BASELINE COMPARISON")
    print("=" * 55)
    print(comp_df.to_string(index=False))

    # -----------------------------------------------------------------------
    # Save training history
    # -----------------------------------------------------------------------
    history_path = os.path.join(METRICS_DIR, "bilstm_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved → {history_path}")

    # Plot training curves
    epochs_ran = list(range(1, len(history["train_loss"]) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs_ran, history["train_loss"], label="Train Loss",  marker="o", markersize=3)
    ax1.plot(epochs_ran, history["val_loss"],   label="Val Loss",    marker="o", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Bi-LSTM Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_ran, history["val_f1"], label="Val F1 (macro)", color="green", marker="o", markersize=3)
    ax2.axhline(0.611, color="red", linestyle="--", label="NB Baseline (0.611)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score (macro)")
    ax2.set_title("Bi-LSTM Validation F1 vs Naive Bayes Baseline")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    curves_path = os.path.join(FIGURES_DIR, "bilstm_training_curves.png")
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f"Training curves saved → {curves_path}")

    # -----------------------------------------------------------------------
    # Final verdict
    # -----------------------------------------------------------------------
    nb_baseline = 0.611
    print("\n" + "=" * 55)
    if f1_macro > nb_baseline:
        print(f"RESULT: Bi-LSTM BEATS Naive Bayes baseline!")
        print(f"  Bi-LSTM F1_macro = {f1_macro:.4f}  vs  NB baseline = {nb_baseline:.3f}")
        print(f"  Improvement: +{f1_macro - nb_baseline:.4f}")
    else:
        print(f"RESULT: Bi-LSTM does NOT beat Naive Bayes baseline.")
        print(f"  Bi-LSTM F1_macro = {f1_macro:.4f}  vs  NB baseline = {nb_baseline:.3f}")
        print(f"  Gap: {nb_baseline - f1_macro:.4f}")
    print("=" * 55)

    return f1_macro


if __name__ == "__main__":
    main()
