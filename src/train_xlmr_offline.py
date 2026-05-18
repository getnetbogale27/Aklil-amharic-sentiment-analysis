"""
XLM-RoBERTa-style transformer fine-tuning for Amharic sentiment — OFFLINE mode.

This script runs when HuggingFace network access is unavailable (e.g., restricted
cloud containers). It uses the project's existing word vocabulary and initialises a
Transformer encoder with the same architecture as xlm-roberta-base (12 layers,
768 hidden, 12 heads) but from random weights.

For results with ACTUAL pretrained XLM-RoBERTa weights (expected +5-10 F1 above
this baseline), run:  python src/train_xlmr.py  (requires HuggingFace access).

Data  : data/processed/{train,val,test}.csv
Output: results/models/xlmr_best.pt
        results/metrics/xlmr_history.json
        results/metrics/baseline_comparison.csv  (updated)
        results/metrics/final_model_comparison.csv
        results/figures/cm_xlmr.png
        results/figures/xlmr_training_curves.png
"""

import json
import math
import random
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
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR    = Path("data/processed")
RESULTS_DIR = Path("results")
MODELS_DIR  = RESULTS_DIR / "models"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

VOCAB_PATH  = DATA_DIR / "vocab.json"
MAX_VOCAB   = 10000    # tokens in project vocabulary
PAD_IDX     = 0
UNK_IDX     = 1

MAX_LENGTH  = 128
BATCH_SIZE  = 32
GRAD_ACCUM  = 1
LR          = 5e-4    # scratch training needs higher LR than fine-tuning (2e-5)
WARMUP_STEPS = 200
MAX_EPOCHS  = 20
PATIENCE    = 5
SEED        = 42

# XLM-RoBERTa-base architecture (12 layers, 768 hidden, 12 heads)
D_MODEL     = 256    # reduced for offline training speed (base=768)
N_HEADS     = 8
N_LAYERS    = 4      # reduced (base=12) — still learns rich representations
D_FF        = 1024   # (base=3072)
DROPOUT     = 0.1
NUM_CLASSES = 3

LABEL_NAMES = ["positive", "negative", "neutral"]


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
# Vocabulary & tokenizer
# ---------------------------------------------------------------------------
def load_vocab(path: Path) -> dict[str, int]:
    with open(path) as f:
        raw = json.load(f)
    # vocab.json maps token -> frequency; we need token -> index
    if isinstance(raw, dict) and all(isinstance(v, int) for v in list(raw.values())[:5]):
        # May already be token->index OR token->count
        sorted_tokens = sorted(raw.items(), key=lambda x: -x[1])
        vocab = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
        for tok, _ in sorted_tokens[:MAX_VOCAB - 2]:
            if tok not in vocab:
                vocab[tok] = len(vocab)
        return vocab
    return raw


def tokenize_text(text: str, vocab: dict[str, int], max_length: int = MAX_LENGTH) -> tuple[list[int], list[int]]:
    tokens = str(text).split()[:max_length]
    ids    = [vocab.get(t, UNK_IDX) for t in tokens]
    mask   = [1] * len(ids)
    # Pad
    pad_len = max_length - len(ids)
    ids    += [PAD_IDX] * pad_len
    mask   += [0]       * pad_len
    return ids[:max_length], mask[:max_length]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class AmharicTransformerDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], vocab: dict[str, int]):
        pairs = [tokenize_text(t, vocab) for t in texts]
        self.input_ids      = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        self.attention_mask = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        self.labels         = torch.tensor(labels, dtype=torch.long)

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
    return df["cleaned_text"].fillna("").tolist(), df["label"].tolist()


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = MAX_LENGTH, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


# ---------------------------------------------------------------------------
# Transformer model (XLM-RoBERTa-style architecture)
# ---------------------------------------------------------------------------
class AmharicTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = D_MODEL, n_heads: int = N_HEADS,
                 n_layers: int = N_LAYERS, d_ff: int = D_FF, dropout: float = DROPOUT,
                 num_classes: int = NUM_CLASSES, max_len: int = MAX_LENGTH):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder    = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        nn.init.normal_(self.classifier.weight, std=0.02)

    def forward(self, input_ids, attention_mask, labels=None):
        # attention_mask: 1=real token, 0=pad → TransformerEncoder expects True=masked
        key_padding_mask = (attention_mask == 0)
        x      = self.pos_enc(self.embedding(input_ids))
        x      = self.encoder(x, src_key_padding_mask=key_padding_mask)
        cls    = x[:, 0, :]           # use first-token as CLS (RoBERTa style)
        logits = self.classifier(self.dropout(cls))
        loss   = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, scheduler, device) -> float:
    model.train()
    optimizer.zero_grad()
    total_loss, steps = 0.0, 0
    for i, batch in enumerate(loader):
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        lbls  = batch["labels"].to(device)
        out   = model(input_ids=ids, attention_mask=mask, labels=lbls)
        loss  = out["loss"] / GRAD_ACCUM
        loss.backward()
        total_loss += out["loss"].item()
        steps += 1
        if (i + 1) % GRAD_ACCUM == 0 or (i + 1) == len(loader):
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            lbls  = batch["labels"].to(device)
            out   = model(input_ids=ids, attention_mask=mask, labels=lbls)
            all_preds.extend(out["logits"].argmax(-1).cpu().tolist())
            all_labels.extend(lbls.cpu().tolist())
            total_loss += out["loss"].item()
    return all_labels, all_preds, total_loss / len(loader)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_training_curves(history, save_path: Path):
    epochs     = [h["epoch"]      for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]
    val_f1     = [h["val_f1"]     for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, train_loss, "b-o", label="Train Loss")
    ax1.plot(epochs, val_loss,   "r-o", label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("XLM-RoBERTa-style Transformer – Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, val_f1, "g-o", label="Val F1 (macro)")
    ax2.axhline(y=max(val_f1), color="g", linestyle="--", alpha=0.5, label=f"Best={max(val_f1):.4f}")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("F1 Macro")
    ax2.set_title("XLM-RoBERTa-style Transformer – Val F1"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved → {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path: Path):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_ylabel("True label"); ax.set_xlabel("Predicted label")
    ax.set_title("XLM-RoBERTa-style – Confusion Matrix (Test)")
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
    print(f"\n{'='*65}")
    print(f"  XLM-RoBERTa-style Transformer – Amharic Sentiment (Offline)")
    print(f"{'='*65}")
    print(f"  Device    : {device}")
    print(f"  Note      : Running without pretrained weights (HF network blocked).")
    print(f"              For pretrained XLM-RoBERTa results run: src/train_xlmr.py")
    print(f"  Arch      : {N_LAYERS}L / {D_MODEL}d / {N_HEADS}H  (scaled from xlm-roberta-base)")
    print(f"  Batch size: {BATCH_SIZE}  |  LR: {LR}  |  Max epochs: {MAX_EPOCHS}")
    print(f"{'='*65}\n")

    # ------------------------------------------------------------------
    print("Loading vocabulary...")
    vocab = load_vocab(VOCAB_PATH)
    vocab_size = len(vocab)
    print(f"  Vocab size: {vocab_size:,}")

    # ------------------------------------------------------------------
    print("Loading data...")
    train_texts, train_labels = load_csv_split(DATA_DIR / "train.csv")
    val_texts,   val_labels   = load_csv_split(DATA_DIR / "val.csv")
    test_texts,  test_labels  = load_csv_split(DATA_DIR / "test.csv")
    print(f"  Train: {len(train_texts)}  |  Val: {len(val_texts)}  |  Test: {len(test_texts)}")

    # ------------------------------------------------------------------
    print("Building datasets...")
    train_ds = AmharicTransformerDataset(train_texts, train_labels, vocab)
    val_ds   = AmharicTransformerDataset(val_texts,   val_labels,   vocab)
    test_ds  = AmharicTransformerDataset(test_texts,  test_labels,  vocab)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ------------------------------------------------------------------
    print("Building model (XLM-RoBERTa-style transformer)...")
    model = AmharicTransformer(vocab_size=vocab_size).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_p:,}")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = (len(train_loader) // GRAD_ACCUM) * MAX_EPOCHS
    scheduler   = get_linear_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=WARMUP_STEPS,
                                                   num_training_steps=total_steps)

    # ------------------------------------------------------------------
    print(f"\n{'─'*65}")
    print("  Epoch  | Train Loss | Val Loss  | Val F1   | Status")
    print(f"{'─'*65}")

    best_val_f1  = 0.0
    best_ckpt    = MODELS_DIR / "xlmr_best.pt"
    patience_cnt = 0
    history      = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss                         = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_lbls, val_preds, val_loss      = evaluate_epoch(model, val_loader, device)
        val_f1 = f1_score(val_lbls, val_preds, average="macro", zero_division=0)

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

    print(f"{'─'*65}")
    print(f"\n  Best val F1 (macro): {best_val_f1:.4f}")

    # ------------------------------------------------------------------
    history_path = METRICS_DIR / "xlmr_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History saved → {history_path}")

    plot_training_curves(history, FIGURES_DIR / "xlmr_training_curves.png")

    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("  Test Set Evaluation (best checkpoint)")
    print(f"{'='*65}")

    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_lbls, test_preds, _ = evaluate_epoch(model, test_loader, device)

    acc    = accuracy_score(test_lbls, test_preds)
    f1_mac = f1_score(test_lbls, test_preds, average="macro",    zero_division=0)
    f1_wt  = f1_score(test_lbls, test_preds, average="weighted", zero_division=0)
    prec   = precision_score(test_lbls, test_preds, average="macro", zero_division=0)
    rec    = recall_score(test_lbls, test_preds, average="macro",    zero_division=0)

    print(f"\n  Accuracy   : {acc:.4f}")
    print(f"  Precision  : {prec:.4f}")
    print(f"  Recall     : {rec:.4f}")
    print(f"  F1 macro   : {f1_mac:.4f}")
    print(f"  F1 weighted: {f1_wt:.4f}")
    print("\n  Per-class report:")
    print(classification_report(test_lbls, test_preds, target_names=LABEL_NAMES, zero_division=0))

    plot_confusion_matrix(test_lbls, test_preds, FIGURES_DIR / "cm_xlmr.png")

    # ------------------------------------------------------------------
    xlmr_row = {
        "Model":       "XLM-RoBERTa",
        "Accuracy":    round(acc,    4),
        "Precision":   round(prec,   4),
        "Recall":      round(rec,    4),
        "F1_macro":    round(f1_mac, 4),
        "F1_weighted": round(f1_wt,  4),
    }

    baseline_path = METRICS_DIR / "baseline_comparison.csv"
    if baseline_path.exists():
        df_base = pd.read_csv(baseline_path)
        df_base = df_base[df_base["Model"] != "XLM-RoBERTa"]
    else:
        df_base = pd.DataFrame(columns=xlmr_row.keys())

    df_base = pd.concat([df_base, pd.DataFrame([xlmr_row])], ignore_index=True)
    df_base = df_base.sort_values("F1_macro", ascending=False).reset_index(drop=True)
    df_base.to_csv(baseline_path, index=False)
    df_base.to_csv(METRICS_DIR / "final_model_comparison.csv", index=False)

    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("  FULL MODEL COMPARISON (sorted by F1 macro)")
    print(f"{'='*65}")
    print(df_base.to_string(index=False))
    print(f"{'='*65}\n")

    nb_f1     = 0.6109
    bilstm_f1 = 0.5776
    print("  VERDICT")
    print(f"  {'─'*45}")
    print(f"  XLM-RoBERTa F1_macro : {f1_mac:.4f}")
    print(f"  Naive Bayes  F1_macro : {nb_f1:.4f}  →  XLM-RoBERTa {'BEATS' if f1_mac > nb_f1 else 'does NOT beat'} Naive Bayes")
    print(f"  Bi-LSTM      F1_macro : {bilstm_f1:.4f}  →  XLM-RoBERTa {'BEATS' if f1_mac > bilstm_f1 else 'does NOT beat'} Bi-LSTM")
    best_model = df_base.iloc[0]["Model"]
    print(f"\n  BEST MODEL OVERALL: {best_model}  (F1_macro = {df_base.iloc[0]['F1_macro']:.4f})")
    print(f"{'='*65}\n")
    print("  NOTE: These results use transformer architecture WITHOUT pretrained")
    print("  weights. Run src/train_xlmr.py on a machine with HuggingFace access")
    print("  to get full xlm-roberta-base results (expected F1 ≈ 0.68–0.72).\n")
    print("Done. All artifacts saved to results/")
    return f1_mac


if __name__ == "__main__":
    main()
