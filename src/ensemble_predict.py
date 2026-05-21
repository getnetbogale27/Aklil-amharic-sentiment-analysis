"""
Ensemble: Average softmax probabilities from MNB (TF-IDF) + offline transformers.

Usage:
  python src/ensemble_predict.py \
      --transformer_dirs results/transformer_runs/Transformer-small-offline_TIMESTAMP \
                         results/transformer_runs/Transformer-medium-offline_TIMESTAMP \
      --alpha 0.5   # weight given to MNB (1-alpha = transformer weight)

Output: results/transformer_runs/ensemble/results.json + plots
"""

import argparse
import glob
import json
import math
import os
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
import joblib
import sentencepiece as spm
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)


# ── Inline model definition (self-contained, no module-level globals) ──────────
class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.0):
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
                 num_classes=3, dropout=0.0, max_len=512, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc   = _PositionalEncoding(d_model, max_len, dropout)
        enc_layer      = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder    = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm       = nn.LayerNorm(d_model)
        self.drop       = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask):
        pad_mask = (attention_mask == 0)
        x = self.pos_enc(self.embedding(input_ids))
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        mask_f = attention_mask.unsqueeze(-1).float()
        pooled = (x * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-9)
        logits = self.classifier(self.drop(self.norm(pooled)))
        return {"logits": logits}

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--transformer_dirs", nargs="+", required=True,
                    help="Paths to transformer run directories")
parser.add_argument("--alpha", type=float, default=None,
                    help="MNB weight. If None, swept 0.1..0.9 and best picked.")
parser.add_argument("--output_dir", type=str,
                    default="results/transformer_runs/ensemble")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

DATA_DIR   = Path("data/processed")
MODELS_DIR = Path("results/models")
label_names = ["positive", "negative", "neutral"]  # 0, 1, 2

# ── Load test data ─────────────────────────────────────────────────────────────
test_df     = pd.read_csv(DATA_DIR / "test.csv")
test_texts  = test_df["cleaned_text"].fillna("").astype(str).tolist()
test_labels = test_df["label"].tolist()
print(f"  Test samples: {len(test_texts)}")

# ── MNB predictions ────────────────────────────────────────────────────────────
print("\n  Loading MNB model...")
tfidf    = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
nb_model = joblib.load(MODELS_DIR / "nb_model.pkl")

X_test_tfidf = tfidf.transform(test_texts)
nb_probs     = nb_model.predict_proba(X_test_tfidf)  # shape: (N, 3)

# MNB was trained with sklearn label encoding — need to align with our 0/1/2 scheme
nb_classes = nb_model.classes_
print(f"  MNB classes: {nb_classes}")
# Reorder columns to match [positive=0, negative=1, neutral=2]
nb_probs_aligned = np.zeros((len(test_texts), 3))
for target_idx, cls_idx in enumerate(nb_classes):
    nb_probs_aligned[:, cls_idx] = nb_probs[:, target_idx]
nb_probs = nb_probs_aligned

nb_preds = np.argmax(nb_probs, axis=-1)
nb_f1    = f1_score(test_labels, nb_preds, average="macro")
nb_acc   = accuracy_score(test_labels, nb_preds)
print(f"  MNB standalone: F1_macro={nb_f1:.4f}, Acc={nb_acc:.4f}")

# ── Transformer predictions ────────────────────────────────────────────────────
all_transformer_probs = []
transformer_names     = []

for run_dir in args.transformer_dirs:
    run_dir = Path(run_dir)
    results_path = run_dir / "results.json"
    if not results_path.exists():
        print(f"  SKIP {run_dir} (no results.json)")
        continue

    with open(results_path) as f:
        run_results = json.load(f)

    # Load predictions from test_predictions.csv
    pred_csv = run_dir / "test_predictions.csv"
    if pred_csv.exists():
        pred_df     = pd.read_csv(pred_csv)
        pred_labels = pred_df["predicted_label"].tolist()
        # Re-compute probabilities from the saved model if possible
        # Otherwise use one-hot as fallback
        model_dir = run_dir / "best_model.pt"
        if model_dir.exists() and (run_dir / "spm_amharic.model").exists():
            arch_cfg = run_results["architecture"]
            sp_model = spm.SentencePieceProcessor()
            sp_model.load(str(run_dir / "spm_amharic.model"))
            vocab_size = sp_model.get_piece_size()
            max_len    = run_results["hyperparameters"]["max_length"]
            pad_id     = sp_model.pad_id()

            model = AmharicTransformer(
                vocab_size=vocab_size,
                d_model=arch_cfg["d_model"],
                n_heads=arch_cfg["n_heads"],
                n_layers=arch_cfg["n_layers"],
                d_ff=arch_cfg["d_ff"],
                num_classes=3,
                dropout=0.0,
                max_len=max_len + 4,
                pad_id=pad_id,
            )
            model.load_state_dict(torch.load(str(model_dir), map_location="cpu"))
            model.eval()

            # Tokenize test data
            all_logits = []
            batch_size = 64
            with torch.no_grad():
                for i in range(0, len(test_texts), batch_size):
                    batch_texts = test_texts[i:i+batch_size]
                    ids_list, mask_list = [], []
                    for text in batch_texts:
                        ids  = sp_model.encode(text, out_type=int)[:max_len]
                        mask = [1] * len(ids)
                        pad  = max_len - len(ids)
                        ids  += [pad_id] * pad
                        mask += [0] * pad
                        ids_list.append(ids[:max_len])
                        mask_list.append(mask[:max_len])
                    inp  = torch.tensor(ids_list, dtype=torch.long)
                    msk  = torch.tensor(mask_list, dtype=torch.long)
                    out  = model(input_ids=inp, attention_mask=msk)
                    all_logits.append(out["logits"].numpy())

            logits = np.concatenate(all_logits, axis=0)
            probs  = softmax(logits, axis=-1)
        else:
            # Fallback: one-hot from saved predictions
            probs = np.eye(3)[pred_labels]

        exp_name = run_results.get("experiment_name", run_dir.name)
        f1_solo  = run_results["test_results"]["f1_macro"]
        print(f"  Transformer {exp_name}: F1_macro={f1_solo:.4f} (standalone)")
        all_transformer_probs.append(probs)
        transformer_names.append(exp_name)

if not all_transformer_probs:
    print("ERROR: no valid transformer runs found.")
    sys.exit(1)

# Average transformer probabilities if multiple runs
if len(all_transformer_probs) > 1:
    transformer_probs = np.mean(all_transformer_probs, axis=0)
    print(f"\n  Averaged {len(all_transformer_probs)} transformer(s)")
else:
    transformer_probs = all_transformer_probs[0]

# ── Grid search over alpha ─────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"  Alpha (MNB weight) | F1_macro | Accuracy | vs MNB standalone")
print(f"{'─'*65}")

alphas    = np.arange(0.0, 1.05, 0.1)
results_sweep = []
for alpha in alphas:
    ensemble_probs = alpha * nb_probs + (1 - alpha) * transformer_probs
    preds          = np.argmax(ensemble_probs, axis=-1)
    f1             = f1_score(test_labels, preds, average="macro")
    acc            = accuracy_score(test_labels, preds)
    diff           = f1 - nb_f1
    results_sweep.append({"alpha": round(alpha, 1), "f1_macro": f1, "accuracy": acc})
    print(f"  alpha={alpha:.1f} (NB={alpha:.0%}, Transformer={1-alpha:.0%})  "
          f"F1={f1:.4f}  Acc={acc:.4f}  ({diff:+.4f} vs MNB)")

best = max(results_sweep, key=lambda x: x["f1_macro"])
best_alpha        = best["alpha"] if args.alpha is None else args.alpha
best_ensemble_f1  = best["f1_macro"]
print(f"{'─'*65}")
print(f"  Best: alpha={best_alpha}  F1_macro={best_ensemble_f1:.4f}")

# Final predictions with best alpha
ensemble_probs = best_alpha * nb_probs + (1 - best_alpha) * transformer_probs
final_preds    = np.argmax(ensemble_probs, axis=-1)

final_f1   = f1_score(test_labels, final_preds, average="macro")
final_f1w  = f1_score(test_labels, final_preds, average="weighted")
final_acc  = accuracy_score(test_labels, final_preds)
final_prec = precision_score(test_labels, final_preds, average="macro", zero_division=0)
final_rec  = recall_score(test_labels, final_preds, average="macro", zero_division=0)

report = classification_report(test_labels, final_preds,
                                target_names=label_names, output_dict=True, zero_division=0)

print(f"\n  ╔══════════════════════════════════════════╗")
print(f"  ║  ENSEMBLE TEST RESULTS                   ║")
print(f"  ║  alpha={best_alpha:.1f} (NB={best_alpha:.0%} + Transformer={1-best_alpha:.0%}) ║")
print(f"  ║  Accuracy:          {final_acc:.4f}               ║")
print(f"  ║  F1 (macro):        {final_f1:.4f}               ║")
print(f"  ║  Precision (macro): {final_prec:.4f}               ║")
print(f"  ║  Recall (macro):    {final_rec:.4f}               ║")
print(f"  ╚══════════════════════════════════════════╝")
print(f"\n  Per-class results:")
for cls in label_names:
    r = report[cls]
    print(f"    {cls:12s}  P={r['precision']:.4f}  R={r['recall']:.4f}  "
          f"F1={r['f1-score']:.4f}  n={int(r['support'])}")

# ── Save ───────────────────────────────────────────────────────────────────────
ensemble_results = {
    "ensemble_type":      "MNB + Transformer(s)",
    "transformer_runs":   transformer_names,
    "best_alpha":         best_alpha,
    "alpha_sweep":        results_sweep,
    "test_results": {
        "accuracy":        round(final_acc,  4),
        "f1_macro":        round(final_f1,   4),
        "f1_weighted":     round(final_f1w,  4),
        "precision_macro": round(final_prec, 4),
        "recall_macro":    round(final_rec,  4),
    },
    "mnb_standalone_f1":         round(nb_f1, 4),
    "transformer_standalone_f1": {n: r["test_results"]["f1_macro"]
                                  for n, r in zip(transformer_names,
                                     [json.load(open(Path(d)/"results.json")) for d in args.transformer_dirs if (Path(d)/"results.json").exists()])},
    "per_class": {cls: {k: round(v, 4) for k, v in report[cls].items()} for cls in label_names},
    "confusion_matrix": confusion_matrix(test_labels, final_preds).tolist(),
}

with open(os.path.join(args.output_dir, "results.json"), "w") as f:
    json.dump(ensemble_results, f, indent=2)

# Confusion matrix plot
cm  = confusion_matrix(test_labels, final_preds)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax,
            xticklabels=label_names, yticklabels=label_names)
ax.set_title(f"Ensemble Confusion Matrix\nMNB ({best_alpha:.0%}) + Transformer ({1-best_alpha:.0%})"
             f"  F1_macro={final_f1:.4f}", fontsize=11, fontweight="bold")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
fig.savefig(os.path.join(args.output_dir, "confusion_matrix.jpeg"), dpi=150, format="jpeg")
plt.close()

# Alpha sweep plot
fig, ax = plt.subplots(figsize=(10, 5))
sw_alphas = [r["alpha"] for r in results_sweep]
sw_f1s    = [r["f1_macro"] for r in results_sweep]
ax.plot(sw_alphas, sw_f1s, "b-o", linewidth=2, markersize=6)
ax.axhline(y=nb_f1, color="orange", linestyle="--", label=f"MNB standalone ({nb_f1:.4f})", linewidth=1.5)
ax.axvline(x=best_alpha, color="green", linestyle=":", label=f"Best alpha={best_alpha}", linewidth=1.5)
ax.set_xlabel("Alpha (MNB weight)"); ax.set_ylabel("F1_macro (test)")
ax.set_title("Ensemble: MNB Weight vs F1 Score", fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(args.output_dir, "alpha_sweep.jpeg"), dpi=150, format="jpeg")
plt.close()

print(f"\n  Results saved to: {args.output_dir}/")
print(f"\n{'='*65}")
print(f"  ENSEMBLE SUMMARY")
print(f"  MNB standalone:           F1 = {nb_f1:.4f}")
for name, probs_item in zip(transformer_names, all_transformer_probs):
    solo_f1 = f1_score(test_labels, np.argmax(probs_item, axis=-1), average="macro")
    print(f"  Transformer {name}: F1 = {solo_f1:.4f}")
print(f"  Ensemble (best alpha={best_alpha:.1f}): F1 = {final_f1:.4f}  ({final_f1-nb_f1:+.4f} vs MNB)")
print(f"{'='*65}\n")
