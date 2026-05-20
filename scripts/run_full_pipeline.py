"""
Full end-to-end pipeline for Aklil Amharic Sentiment Analysis.
Runs EDA, ML baselines, loads Bi-LSTM history, checks XLM-R,
and exports all outputs under results/run_YYYYMMDD_HHMMSS/.
"""

import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
RAW_DIR  = ROOT / "data" / "raw" / "amh"
RESULTS  = ROOT / "results"

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = RESULTS / f"run_{TS}"
TABLES_DIR = RUN_DIR / "tables"
PLOTS_DIR  = RUN_DIR / "plots"

for d in (TABLES_DIR, PLOTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

print(f"\n{'='*65}")
print(f"  Aklil Amharic Sentiment Analysis — Full Pipeline Run")
print(f"  Output folder: {RUN_DIR}")
print(f"  Timestamp    : {TS}")
print(f"{'='*65}\n")

LABEL_NAMES   = ["positive", "negative", "neutral"]
LABEL_MAP_STR = {0: "positive", 1: "negative", 2: "neutral"}
LABEL_COLORS  = {"positive": "#4CAF50", "negative": "#F44336", "neutral": "#2196F3"}

# ── Helpers ────────────────────────────────────────────────────────────────────

def savefig(fig, name):
    path = PLOTS_DIR / f"{name}_{TS}.jpeg"
    fig.savefig(path, dpi=150, bbox_inches="tight", format="jpeg")
    plt.close(fig)
    print(f"  [plot] {path.name}")
    return path


def save_xlsx(df_or_dict, name, sheet="Sheet1"):
    path = TABLES_DIR / f"{name}_{TS}.xlsx"
    if isinstance(df_or_dict, dict):
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for sh, df in df_or_dict.items():
                df.to_excel(writer, sheet_name=sh, index=False)
    else:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df_or_dict.to_excel(writer, sheet_name=sheet, index=False)
    print(f"  [xlsx] {path.name}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load Data & EDA
# ═══════════════════════════════════════════════════════════════════════════════
print("─" * 65)
print("STEP 1 — Load Data & EDA")
print("─" * 65)

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df   = pd.read_csv(DATA_DIR / "val.csv")
test_df  = pd.read_csv(DATA_DIR / "test.csv")
full_df  = pd.read_csv(DATA_DIR / "cleaned_dataset.csv")

# Normalize label column: might be int or str
for df in (train_df, val_df, test_df, full_df):
    if "label" not in df.columns and "label_str" in df.columns:
        df["label"] = df["label_str"].map({"positive": 0, "negative": 1, "neutral": 2})

n_total = len(full_df)
n_train = len(train_df)
n_val   = len(val_df)
n_test  = len(test_df)

print(f"  Total samples : {n_total:,}")
print(f"  Train / Val / Test : {n_train} / {n_val} / {n_test}")

# Label distribution
label_counts = full_df["label"].value_counts().sort_index()
label_dist   = {LABEL_MAP_STR[k]: int(v) for k, v in label_counts.items()}
print(f"  Label dist    : {label_dist}")

# Tweet length (character count of cleaned_text)
full_df["text_len"] = full_df["cleaned_text"].fillna("").str.len()

# ── EDA: label distribution bar chart ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
labels_ordered = ["positive", "negative", "neutral"]
counts = [label_dist.get(l, 0) for l in labels_ordered]
colors = [LABEL_COLORS[l] for l in labels_ordered]
bars = ax.bar(labels_ordered, counts, color=colors, edgecolor="white", linewidth=1.2)
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
            f"{cnt:,}\n({cnt/n_total*100:.1f}%)", ha="center", va="bottom", fontsize=11)
ax.set_title("Sentiment Label Distribution — AfriSenti Amharic", fontsize=13, fontweight="bold")
ax.set_xlabel("Sentiment Class", fontsize=11)
ax.set_ylabel("Number of Tweets", fontsize=11)
ax.set_ylim(0, max(counts) * 1.22)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3)
savefig(fig, "label_distribution")

# ── EDA: tweet length distribution by sentiment ────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for lbl_idx, lbl_name in enumerate(labels_ordered):
    subset = full_df[full_df["label"] == lbl_idx]["text_len"]
    ax.hist(subset, bins=40, alpha=0.6, color=LABEL_COLORS[lbl_name],
            label=f"{lbl_name} (n={len(subset):,})", edgecolor="none")
ax.set_title("Tweet Length Distribution by Sentiment Class", fontsize=13, fontweight="bold")
ax.set_xlabel("Character Count (cleaned text)", fontsize=11)
ax.set_ylabel("Frequency", fontsize=11)
ax.legend(fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3)
savefig(fig, "tweet_length_distribution")

# ── EDA: dataset summary Excel ─────────────────────────────────────────────────
summary_rows = []
for lbl_idx, lbl_name in enumerate(labels_ordered):
    subset = full_df[full_df["label"] == lbl_idx]["text_len"]
    summary_rows.append({
        "sentiment_class": lbl_name,
        "count":           int(len(subset)),
        "percentage":      round(len(subset) / n_total * 100, 2),
        "avg_length":      round(float(subset.mean()), 2),
        "min_length":      int(subset.min()),
        "max_length":      int(subset.max()),
    })
summary_df = pd.DataFrame(summary_rows)
save_xlsx(summary_df, "dataset_summary")
print("\n  Dataset Summary:")
print(summary_df.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Baseline ML Models
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("STEP 2 — Baseline ML Models (TF-IDF + 4 classifiers)")
print("─" * 65)

TEXT_COL  = "cleaned_text"
LABEL_COL = "label"

def load_split_df(df):
    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).copy()
    return df[TEXT_COL].astype(str).tolist(), df[LABEL_COL].astype(int).tolist()

X_train, y_train = load_split_df(train_df)
X_val,   y_val   = load_split_df(val_df)
X_test,  y_test  = load_split_df(test_df)

print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf   = tfidf.transform(X_val)
X_test_tfidf  = tfidf.transform(X_test)
print(f"  TF-IDF vocabulary size: {len(tfidf.vocabulary_):,}")

MODELS = [
    ("Naive Bayes",         MultinomialNB(),                          "naive_bayes"),
    ("Logistic Regression", LogisticRegression(max_iter=1000, C=1.0), "logistic_regression"),
    ("SVM",                 LinearSVC(max_iter=2000),                 "svm"),
    ("KNN",                 KNeighborsClassifier(n_neighbors=5),      "knn"),
]

comparison_rows  = []
per_class_sheets = {}

for model_name, clf, slug in MODELS:
    print(f"\n  Training {model_name} …")
    t0 = time.time()
    clf.fit(X_train_tfidf, y_train)
    train_sec = round(time.time() - t0, 3)

    y_pred = clf.predict(X_test_tfidf)

    acc    = accuracy_score(y_test, y_pred)
    prec   = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec    = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1mac  = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"    Acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1mac:.4f}  t={train_sec}s")

    comparison_rows.append({
        "Model":              model_name,
        "Accuracy":           round(acc,    4),
        "Precision_macro":    round(prec,   4),
        "Recall_macro":       round(rec,    4),
        "F1_macro":           round(f1mac,  4),
        "Training_time_sec":  train_sec,
    })

    # Per-class classification report
    report = classification_report(y_test, y_pred, target_names=LABEL_NAMES,
                                   zero_division=0, output_dict=True)
    rows = []
    for cls in LABEL_NAMES + ["macro avg", "weighted avg"]:
        if cls in report:
            r = report[cls]
            rows.append({
                "Class":     cls,
                "Precision": round(r["precision"], 4),
                "Recall":    round(r["recall"],    4),
                "F1-score":  round(r["f1-score"],  4),
                "Support":   int(r["support"]),
            })
    per_class_sheets[model_name] = pd.DataFrame(rows)

    # Confusion matrix heatmap
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, f"cm_{slug}")

# Save per-class reports
save_xlsx(per_class_sheets, "per_class_reports")

# Save baseline comparison
baseline_df = (pd.DataFrame(comparison_rows)
               .sort_values("F1_macro", ascending=False)
               .reset_index(drop=True))
save_xlsx(baseline_df, "ml_baseline_comparison")

# Grouped bar chart: Accuracy vs F1_macro
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(comparison_rows))
w = 0.35
ax.bar(x - w/2, [r["Accuracy"]  for r in comparison_rows], w, label="Accuracy",  color="#5C85D6", edgecolor="white")
ax.bar(x + w/2, [r["F1_macro"]  for r in comparison_rows], w, label="F1 macro",  color="#E07B54", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([r["Model"] for r in comparison_rows], fontsize=11)
ax.set_ylim(0, 0.85)
ax.set_title("ML Baseline Comparison — Accuracy vs F1 Macro", fontsize=13, fontweight="bold")
ax.set_ylabel("Score", fontsize=11)
ax.legend(fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3)
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=8.5)
savefig(fig, "ml_baseline_comparison")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Bi-LSTM (load existing history)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("STEP 3 — Bi-LSTM (load saved history + metrics)")
print("─" * 65)

history_path  = RESULTS / "metrics" / "bilstm_history.json"
bilstm_f1mac  = None

if history_path.exists():
    with open(history_path) as f:
        hist = json.load(f)

    epochs = list(range(1, len(hist["train_loss"]) + 1))
    best_epoch = int(np.argmax(hist["val_f1"])) + 1
    best_val_f1 = float(max(hist["val_f1"]))
    print(f"  Loaded {len(epochs)} epochs. Best val F1: {best_val_f1:.4f} at epoch {best_epoch}")

    # Read actual test F1 from saved comparison
    old_comp = RESULTS / "metrics" / "baseline_comparison.csv"
    if old_comp.exists():
        odf = pd.read_csv(old_comp)
        row = odf[odf["Model"] == "Bi-LSTM"]
        if not row.empty:
            bilstm_f1mac  = float(row["F1_macro"].iloc[0])
            bilstm_acc    = float(row["Accuracy"].iloc[0])
            bilstm_prec   = float(row["Precision"].iloc[0])
            bilstm_rec    = float(row["Recall"].iloc[0])
            print(f"  Bi-LSTM test F1_macro: {bilstm_f1mac:.4f}  Acc: {bilstm_acc:.4f}")

    # Training loss curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(epochs, hist["train_loss"], "o-", markersize=3, label="Train Loss",  color="#5C85D6")
    ax1.plot(epochs, hist["val_loss"],   "s-", markersize=3, label="Val Loss",    color="#E07B54")
    ax1.axvline(best_epoch, color="gray", linestyle="--", alpha=0.6, label=f"Best epoch ({best_epoch})")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_title("Bi-LSTM Training & Validation Loss", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.plot(epochs, hist["val_f1"], "o-", markersize=3, label="Val F1 macro", color="#4CAF50")
    ax2.axhline(0.6109, color="#E07B54", linestyle="--", linewidth=1.5,
                label="NB Baseline (0.611)")
    ax2.axvline(best_epoch, color="gray", linestyle="--", alpha=0.6, label=f"Best epoch ({best_epoch})")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("F1 Score (macro)", fontsize=11)
    ax2.set_title("Bi-LSTM Validation F1 vs NB Baseline", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    savefig(fig, "bilstm_training_curves")
else:
    print("  [SKIP] No bilstm_history.json found — Bi-LSTM not yet trained.")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — XLM-RoBERTa
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("STEP 4 — XLM-RoBERTa")
print("─" * 65)

xlmr_script = ROOT / "src" / "train_xlmr.py"
xlmr_offline = ROOT / "src" / "train_xlmr_offline.py"
xlmr_history_path = RESULTS / "metrics" / "xlmr_history.json"
xlmr_f1mac = None
xlmr_note  = ""

# Check GPU
try:
    import torch
    gpu_available = torch.cuda.is_available()
    print(f"  PyTorch available. GPU: {gpu_available}")
except ImportError:
    gpu_available = False
    print("  PyTorch not installed — GPU unavailable.")

if xlmr_history_path.exists():
    print("  Found existing xlmr_history.json — loading results.")
    with open(xlmr_history_path) as f:
        xlmr_hist = json.load(f)
    xlmr_f1mac = xlmr_hist.get("test_f1_macro")
    xlmr_note  = "loaded from saved history"
    # plot curves if present
    if "val_f1" in xlmr_hist:
        ep = list(range(1, len(xlmr_hist["val_f1"]) + 1))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ep, xlmr_hist["val_f1"], "o-", markersize=4, color="#9C27B0", label="Val F1")
        if "train_loss" in xlmr_hist:
            ax2 = ax.twinx()
            ax2.plot(ep, xlmr_hist["train_loss"], "s--", markersize=3, color="#FF9800", label="Train Loss")
            ax2.set_ylabel("Train Loss", fontsize=10)
        ax.set_title("XLM-RoBERTa Training Curves", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val F1 macro")
        ax.legend(loc="lower right")
        savefig(fig, "xlmr_training_curves")
elif not gpu_available:
    xlmr_note = "SKIPPED — no GPU available and PyTorch not installed"
    print(f"  [SKIP] {xlmr_note}")
    print("  To train XLM-R, install torch with CUDA and run: python src/train_xlmr.py")
else:
    print("  GPU available but no saved results. Training XLM-R would take several hours.")
    xlmr_note = "SKIPPED — GPU available but training not executed in this run"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Master Comparison Table & Metadata
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("STEP 5 — Master Comparison Table & Run Metadata")
print("─" * 65)

# Build master rows from the freshly-trained baselines + loaded Bi-LSTM
master_rows = list(comparison_rows)  # ML baseline rows already have correct fields

if bilstm_f1mac is not None:
    master_rows.append({
        "Model":             "Bi-LSTM",
        "Accuracy":          bilstm_acc,
        "Precision_macro":   bilstm_prec,
        "Recall_macro":      bilstm_rec,
        "F1_macro":          bilstm_f1mac,
        "Training_time_sec": "N/A (pre-trained)",
    })

if xlmr_f1mac is not None:
    master_rows.append({
        "Model":             "XLM-RoBERTa",
        "Accuracy":          xlmr_hist.get("test_accuracy", "N/A"),
        "Precision_macro":   xlmr_hist.get("test_precision", "N/A"),
        "Recall_macro":      xlmr_hist.get("test_recall", "N/A"),
        "F1_macro":          xlmr_f1mac,
        "Training_time_sec": "N/A (pre-trained)",
    })

master_df = pd.DataFrame(master_rows)
# Sort by F1_macro descending (only numeric rows)
numeric_mask = pd.to_numeric(master_df["F1_macro"], errors="coerce").notna()
master_df = pd.concat([
    master_df[numeric_mask].sort_values("F1_macro", ascending=False),
    master_df[~numeric_mask],
]).reset_index(drop=True)

save_xlsx(master_df, "master_model_comparison")

# Print final ranked summary
print("\n  ╔══════════════════════════════════════════════════════════╗")
print("  ║    FINAL MODEL RANKING (by F1_macro, test set)          ║")
print("  ╠══════════════════════════════════════════════════════════╣")
for i, row in master_df.iterrows():
    f1_str = f"{row['F1_macro']:.4f}" if isinstance(row["F1_macro"], float) else str(row["F1_macro"])
    flag = "  ← BEST" if i == 0 else ""
    print(f"  ║  {i+1}. {row['Model']:<22}  F1_macro = {f1_str}{flag}")
print("  ╚══════════════════════════════════════════════════════════╝")

# Identify best model
best_row = master_df[pd.to_numeric(master_df["F1_macro"], errors="coerce").notna()].iloc[0]
best_model_name = best_row["Model"]
best_f1         = float(best_row["F1_macro"])

# Run metadata JSON
metadata = {
    "timestamp":          TS,
    "total_samples":      n_total,
    "train_size":         n_train,
    "val_size":           n_val,
    "test_size":          n_test,
    "label_distribution": label_dist,
    "tfidf_max_features": 20000,
    "tfidf_ngram_range":  [1, 2],
    "models_evaluated":   [r["Model"] for r in master_rows],
    "best_model":         best_model_name,
    "best_f1_macro":      round(best_f1, 4),
    "xlmr_status":        xlmr_note if xlmr_note else "not evaluated",
    "gpu_available":      gpu_available,
    "python_packages": {
        "scikit-learn": __import__("sklearn").__version__,
        "pandas":       pd.__version__,
        "numpy":        np.__version__,
        "matplotlib":   matplotlib.__version__,
    },
}

meta_path = RUN_DIR / f"run_metadata_{TS}.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"\n  [json] {meta_path.name}")

# ── Final summary ──────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  PIPELINE COMPLETE")
print(f"  Output folder  : {RUN_DIR}")
print(f"  Plots saved    : {len(list(PLOTS_DIR.iterdir()))}")
print(f"  Tables saved   : {len(list(TABLES_DIR.iterdir()))}")
print(f"  Best model     : {best_model_name}  (F1_macro = {best_f1:.4f})")
print(f"{'='*65}\n")

# List all outputs
print("  Files generated:")
for f in sorted(RUN_DIR.rglob("*")):
    if f.is_file():
        rel = f.relative_to(RUN_DIR)
        print(f"    {rel}")
