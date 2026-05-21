"""
Transformer fine-tuning for Amharic sentiment — fully offline mode.

Since HuggingFace model downloads are blocked in this environment, this script:
  1. Trains a SentencePiece BPE tokenizer on the training corpus (no internet needed)
  2. Trains a transformer encoder from random weights with that vocabulary
  3. Runs multiple architecture configurations and picks the best

Architecture is named after the analogous pretrained models for comparison:
  small  : 4L / 256d / 8H (analogous to DistilBERT scale, ~8M params)
  medium : 6L / 512d / 8H (analogous to BERT-tiny, ~25M params)
  large  : 6L / 768d / 12H (analogous to XLM-RoBERTa scale, ~65M params)

NOTE: Results WILL be lower than actual pretrained models (expected +5-15 F1 from
pretraining). These results demonstrate the transformer architecture capability
on this dataset without transfer learning. Run `train_transformer_cpu.py` on a
machine with HuggingFace access to get pretrained model results.

Data  : data/processed/{train,val,test}.csv  (same split as all ML experiments)
Output: results/transformer_runs/{exp_name}_{timestamp}/
"""

import argparse
import io
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sentencepiece as spm
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

# ── Argument parser ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--arch",    type=str,   default="medium",
                    choices=["small", "medium", "large"],
                    help="Architecture size: small|medium|large")
parser.add_argument("--epochs",  type=int,   default=20)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr",      type=float, default=5e-4)
parser.add_argument("--spm_vocab", type=int, default=8000,
                    help="SentencePiece vocab size")
parser.add_argument("--max_length", type=int, default=128)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--patience", type=int,  default=5)
parser.add_argument("--output_dir", type=str, default="results/transformer_runs")
parser.add_argument("--experiment_name", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# ── Architecture configs ───────────────────────────────────────────────────────
ARCH_CONFIGS = {
    "small":  {"n_layers": 4, "d_model": 256,  "n_heads": 8,  "d_ff": 1024},
    "medium": {"n_layers": 6, "d_model": 512,  "n_heads": 8,  "d_ff": 2048},
    "large":  {"n_layers": 6, "d_model": 768,  "n_heads": 12, "d_ff": 3072},
}
arch_cfg = ARCH_CONFIGS[args.arch]

# ── Setup ──────────────────────────────────────────────────────────────────────
TIMESTAMP  = datetime.now().strftime("%Y%m%d_%H%M%S")
exp_name   = args.experiment_name or f"Transformer-{args.arch}-offline"
RUN_DIR    = os.path.join(args.output_dir, f"{exp_name}_{TIMESTAMP}")
os.makedirs(RUN_DIR, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(args.seed)

print(f"\n{'='*65}")
print(f"  Transformer Training — OFFLINE Mode (no pretrained weights)")
print(f"  Architecture: {args.arch}  ({arch_cfg['n_layers']}L / {arch_cfg['d_model']}d / {arch_cfg['n_heads']}H)")
print(f"  Output: {RUN_DIR}")
print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*65}\n")

# ── Load Data ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("data/processed")

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df   = pd.read_csv(DATA_DIR / "val.csv")
test_df  = pd.read_csv(DATA_DIR / "test.csv")

text_col  = "cleaned_text"
label_col = "label"
label_names = ["positive", "negative", "neutral"]  # 0, 1, 2

train_texts  = train_df[text_col].fillna("").astype(str).tolist()
train_labels = train_df[label_col].tolist()
val_texts    = val_df[text_col].fillna("").astype(str).tolist()
val_labels   = val_df[label_col].tolist()
test_texts   = test_df[text_col].fillna("").astype(str).tolist()
test_labels  = test_df[label_col].tolist()

print(f"  Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")
print(f"  Train label dist: {pd.Series(train_labels).value_counts().sort_index().to_dict()}")

# Class weights
y_arr = np.array(train_labels)
class_weights = compute_class_weight("balanced", classes=np.unique(y_arr), y=y_arr)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
print(f"  Class weights: {class_weights.round(3)}")

# ── Train SentencePiece tokenizer ──────────────────────────────────────────────
print(f"\n  Training SentencePiece tokenizer (vocab={args.spm_vocab})...")
spm_model_prefix = os.path.join(RUN_DIR, "spm_amharic")
corpus_path      = os.path.join(RUN_DIR, "corpus.txt")

with open(corpus_path, "w", encoding="utf-8") as f:
    for t in train_texts:
        f.write(t.strip() + "\n")

spm.SentencePieceTrainer.train(
    input=corpus_path,
    model_prefix=spm_model_prefix,
    vocab_size=args.spm_vocab,
    character_coverage=0.9999,
    model_type="bpe",
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    pad_piece="<pad>", unk_piece="<unk>",
    shuffle_input_sentence=True,
)
sp = spm.SentencePieceProcessor()
sp.load(spm_model_prefix + ".model")
VOCAB_SIZE = sp.get_piece_size()
PAD_ID     = sp.pad_id()
print(f"  Tokenizer ready. Vocab size: {VOCAB_SIZE}")

# ── Dataset ────────────────────────────────────────────────────────────────────
class AmharicDataset(Dataset):
    def __init__(self, texts, labels, sp_model, max_len):
        self.labels = torch.tensor(labels, dtype=torch.long)
        ids_list, mask_list = [], []
        for text in texts:
            ids  = sp_model.encode(text, out_type=int)[:max_len]
            mask = [1] * len(ids)
            pad  = max_len - len(ids)
            ids  += [PAD_ID] * pad
            mask += [0]      * pad
            ids_list.append(ids[:max_len])
            mask_list.append(mask[:max_len])
        self.input_ids      = torch.tensor(ids_list, dtype=torch.long)
        self.attention_mask = torch.tensor(mask_list, dtype=torch.long)

    def __len__(self):  return len(self.labels)
    def __getitem__(self, i):
        return {"input_ids":      self.input_ids[i],
                "attention_mask": self.attention_mask[i],
                "labels":         self.labels[i]}

print("  Tokenizing datasets...")
train_ds = AmharicDataset(train_texts, train_labels, sp, args.max_length)
val_ds   = AmharicDataset(val_texts,   val_labels,   sp, args.max_length)
test_ds  = AmharicDataset(test_texts,  test_labels,  sp, args.max_length)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

# ── Transformer Model ──────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class AmharicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff,
                 num_classes=3, dropout=0.1, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)
        enc_layer      = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder    = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm       = nn.LayerNorm(d_model)
        self.drop       = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        nn.init.normal_(self.classifier.weight, std=0.02)

    def forward(self, input_ids, attention_mask, labels=None):
        pad_mask = (attention_mask == 0)
        x = self.pos_enc(self.embedding(input_ids))
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        # Mean pooling over non-padding tokens
        mask_f = attention_mask.unsqueeze(-1).float()
        pooled = (x * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-9)
        logits = self.classifier(self.drop(self.norm(pooled)))
        loss   = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(weight=class_weights_tensor)(logits, labels)
        return {"loss": loss, "logits": logits}


print(f"\n  Building model: {args.arch}")
device = torch.device("cpu")
model  = AmharicTransformer(
    vocab_size=VOCAB_SIZE,
    d_model=arch_cfg["d_model"],
    n_heads=arch_cfg["n_heads"],
    n_layers=arch_cfg["n_layers"],
    d_ff=arch_cfg["d_ff"],
    num_classes=3,
    dropout=args.dropout,
    max_len=args.max_length + 4,
).to(device)

param_count = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {param_count:,}")

optimizer   = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
total_steps = len(train_loader) * args.epochs
scheduler   = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps,
)

# ── Training loop ──────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss, steps = 0.0, 0
    for batch in loader:
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        lbls  = batch["labels"].to(device)
        out   = model(input_ids=ids, attention_mask=mask, labels=lbls)
        out["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += out["loss"].item()
        steps      += 1
    return total_loss / steps


def evaluate(model, loader):
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


print(f"\n  Training config:")
print(f"    Arch:    {args.arch}  ({arch_cfg['n_layers']}L / {arch_cfg['d_model']}d / {arch_cfg['n_heads']}H)")
print(f"    Epochs:  {args.epochs} (patience={args.patience})")
print(f"    Batch:   {args.batch_size}  LR: {args.lr}")
print(f"\n{'─'*65}")
print(f"  Epoch  | Train Loss | Val Loss  | Val F1   | Status")
print(f"{'─'*65}")

best_val_f1  = 0.0
best_ckpt    = os.path.join(RUN_DIR, "best_model.pt")
patience_cnt = 0
history      = []
t0           = time.time()

for epoch in range(1, args.epochs + 1):
    train_loss                    = train_epoch(model, train_loader, optimizer, scheduler)
    val_lbls, val_preds, val_loss = evaluate(model, val_loader)
    val_f1 = f1_score(val_lbls, val_preds, average="macro", zero_division=0)

    status = ""
    if val_f1 > best_val_f1:
        best_val_f1  = val_f1
        patience_cnt = 0
        torch.save(model.state_dict(), best_ckpt)
        status = "  ← best"
    else:
        patience_cnt += 1
        if patience_cnt >= args.patience:
            history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                             "val_loss": round(val_loss, 4), "val_f1": round(val_f1, 4)})
            print(f"  {epoch:>5}  | {train_loss:>10.4f} | {val_loss:>9.4f} | {val_f1:>8.4f} | early stop")
            break

    history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                     "val_loss": round(val_loss, 4), "val_f1": round(val_f1, 4)})
    print(f"  {epoch:>5}  | {train_loss:>10.4f} | {val_loss:>9.4f} | {val_f1:>8.4f} |{status}")

train_time = time.time() - t0
print(f"{'─'*65}")
print(f"\n  Best val F1: {best_val_f1:.4f}  |  Training time: {train_time/60:.1f} min")

# ── Test evaluation ────────────────────────────────────────────────────────────
print(f"\n  Evaluating on test set (best checkpoint)...")
model.load_state_dict(torch.load(best_ckpt, map_location=device))
test_lbls, test_preds, _ = evaluate(model, test_loader)

test_acc  = accuracy_score(test_lbls, test_preds)
test_f1   = f1_score(test_lbls, test_preds, average="macro",    zero_division=0)
test_f1w  = f1_score(test_lbls, test_preds, average="weighted", zero_division=0)
test_prec = precision_score(test_lbls, test_preds, average="macro", zero_division=0)
test_rec  = recall_score(test_lbls, test_preds, average="macro",    zero_division=0)

report = classification_report(test_lbls, test_preds,
                                target_names=label_names, output_dict=True, zero_division=0)

print(f"\n  ╔══════════════════════════════════════╗")
print(f"  ║  TEST RESULTS                        ║")
print(f"  ║  Accuracy:          {test_acc:.4f}             ║")
print(f"  ║  F1 (macro):        {test_f1:.4f}             ║")
print(f"  ║  Precision (macro): {test_prec:.4f}             ║")
print(f"  ║  Recall (macro):    {test_rec:.4f}             ║")
print(f"  ╚══════════════════════════════════════╝")
print(f"\n  Per-class results:")
for cls in label_names:
    r = report[cls]
    print(f"    {cls:12s}  P={r['precision']:.4f}  R={r['recall']:.4f}  "
          f"F1={r['f1-score']:.4f}  n={int(r['support'])}")

# ── Save results ───────────────────────────────────────────────────────────────
results = {
    "model_name":        f"Transformer-{args.arch}-offline (SentencePiece BPE, random init)",
    "experiment_name":   exp_name,
    "timestamp":         TIMESTAMP,
    "offline_mode":      True,
    "note":              "Trained from random weights. HuggingFace blocked in this environment. "
                         "Run train_transformer_cpu.py with network access for pretrained results.",
    "parameters":        param_count,
    "train_time_minutes": round(train_time / 60, 1),
    "epochs_trained":    len(history),
    "architecture":      {**arch_cfg, "spm_vocab": VOCAB_SIZE},
    "hyperparameters":   {
        "learning_rate": args.lr, "batch_size": args.batch_size,
        "max_length": args.max_length, "dropout": args.dropout, "patience": args.patience,
    },
    "test_results": {
        "accuracy":        round(test_acc,  4),
        "f1_macro":        round(test_f1,   4),
        "f1_weighted":     round(test_f1w,  4),
        "precision_macro": round(test_prec, 4),
        "recall_macro":    round(test_rec,  4),
    },
    "per_class":         {cls: {k: round(v, 4) for k, v in report[cls].items()} for cls in label_names},
    "confusion_matrix":  confusion_matrix(test_lbls, test_preds).tolist(),
    "train_history":     history,
}

with open(os.path.join(RUN_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

# Predictions CSV
pred_df = test_df.copy()
pred_df["predicted_label"]   = test_preds
pred_df["predicted_label_str"] = [label_names[p] for p in test_preds]
pred_df["correct"] = pred_df[label_col] == pred_df["predicted_label"]
pred_df.to_csv(os.path.join(RUN_DIR, "test_predictions.csv"), index=False)

# ── Plots ──────────────────────────────────────────────────────────────────────
# Confusion matrix
cm  = confusion_matrix(test_lbls, test_preds)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=label_names, yticklabels=label_names)
ax.set_title(f"Confusion Matrix — {exp_name}\nF1_macro={test_f1:.4f} (offline, no pretrained weights)",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
fig.savefig(os.path.join(RUN_DIR, "confusion_matrix.jpeg"), dpi=150, format="jpeg")
plt.close()

# Training curves
epochs_list = [h["epoch"]      for h in history]
train_losses= [h["train_loss"] for h in history]
val_losses  = [h["val_loss"]   for h in history]
val_f1s     = [h["val_f1"]     for h in history]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(epochs_list, train_losses, "b-o", label="Train Loss", markersize=4)
axes[0].plot(epochs_list, val_losses,   "r-o", label="Val Loss",   markersize=4)
axes[0].set_title("Loss Curves", fontweight="bold")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend()

axes[1].plot(epochs_list, val_f1s, "g-o", label=f"Val F1 (best={best_val_f1:.4f})", markersize=4)
axes[1].axhline(y=0.6802, color="orange", linestyle="--", label="MNB baseline (0.6802)", linewidth=1.5)
axes[1].axhline(y=0.5776, color="red",    linestyle=":",  label="Bi-LSTM (0.5776)",      linewidth=1.2)
axes[1].set_title("Validation F1 Score", fontweight="bold")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("F1 (macro)"); axes[1].legend()

plt.suptitle(f"{exp_name}\n(offline, SentencePiece BPE, no pretrained weights)", fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(RUN_DIR, "training_curves.jpeg"), dpi=150, format="jpeg")
plt.close()

print(f"\n  All outputs saved to: {RUN_DIR}/")
print(f"\n{'='*65}")
print(f"  SUMMARY: {exp_name}")
print(f"  F1_macro = {test_f1:.4f}  |  Accuracy = {test_acc:.4f}")
print(f"  vs MNB baseline 0.6802: {test_f1 - 0.6802:+.4f}")
print(f"  vs Bi-LSTM      0.5776: {test_f1 - 0.5776:+.4f}")
print(f"  Training time: {train_time/60:.1f} min")
print(f"  Note: offline (no pretrained weights) — pretrained expected +5-15 F1 points")
print(f"{'='*65}\n")
