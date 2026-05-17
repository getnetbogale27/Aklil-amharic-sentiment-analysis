"""
EDA script derived from notebooks/01_data_exploration.ipynb.
Saves all plots to results/figures/.
"""

import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing.preprocess import preprocess_text

matplotlib.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
sns.set_theme(style="whitegrid", palette="Set2")

DATA_RAW = Path("data/raw")
DATA_PROC = Path("data/processed")
FIGURES = Path("results/figures")
FIGURES.mkdir(parents=True, exist_ok=True)

# ─── 1. Load AfriSenti splits ─────────────────────────────────────────────────
def load_tsv(path):
    df = pd.read_csv(path, sep="\t", header=None, names=["tweet", "label"],
                     encoding="utf-8")
    df = df[df["label"] != "label"].reset_index(drop=True)
    df["label"] = df["label"].str.strip().str.lower()
    return df[df["label"].isin(["positive", "negative", "neutral"])].reset_index(drop=True)

train_df = load_tsv(DATA_RAW / "amh/train.tsv")
dev_df   = load_tsv(DATA_RAW / "amh/dev.tsv")
test_df  = load_tsv(DATA_RAW / "amh/test.tsv")
print(f"Train: {len(train_df):,}  Dev: {len(dev_df):,}  Test: {len(test_df):,}")

policy_df = pd.read_excel(DATA_RAW / "dataset.xlsx")
policy_df = policy_df.rename(columns={"Text": "tweet", "Sentiment": "label"})
policy_df["label"] = policy_df["label"].str.strip().str.lower()
policy_df = policy_df[policy_df["label"].isin(["positive", "negative", "neutral"])].reset_index(drop=True)
print(f"Policy tweets: {len(policy_df):,}")

# ─── 2. Class distribution ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (df, split) in zip(axes, [(train_df, "Train"), (dev_df, "Dev"), (test_df, "Test")]):
    counts = df["label"].value_counts().reindex(["positive", "negative", "neutral"], fill_value=0)
    bars = ax.bar(counts.index, counts.values, color=["#4CAF50", "#F44336", "#2196F3"])
    ax.set_title(f"{split} ({len(df):,} tweets)")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)
    for bar in bars:
        ax.annotate(f"{int(bar.get_height()):,}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=10)
plt.suptitle("AfriSenti AMH — Sentiment Class Distribution", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES / "class_distribution_afrisenti.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved class_distribution_afrisenti.png")

# ─── 3. Combined dataset class distribution ───────────────────────────────────
combined_proc = pd.read_csv(DATA_PROC / "cleaned_dataset.csv")
label_map = {0: "positive", 1: "negative", 2: "neutral"}
combined_proc["label_str"] = combined_proc["label"].map(label_map)

fig, ax = plt.subplots(figsize=(7, 4))
counts = combined_proc["label_str"].value_counts().reindex(["positive", "negative", "neutral"])
bars = ax.bar(counts.index, counts.values, color=["#4CAF50", "#F44336", "#2196F3"], width=0.5)
ax.set_title(f"Combined Dataset — Label Distribution (n={len(combined_proc):,})")
ax.set_xlabel("Sentiment")
ax.set_ylabel("Count")
for bar in bars:
    h = int(bar.get_height())
    ax.annotate(f"{h:,}\n({100*h/len(combined_proc):.1f}%)",
                (bar.get_x() + bar.get_width() / 2, h),
                ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig(FIGURES / "class_distribution_combined.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved class_distribution_combined.png")

# ─── 4. Tweet length distribution ────────────────────────────────────────────
train_df["char_len"] = train_df["tweet"].str.len()
train_df["word_len"] = train_df["tweet"].str.split().str.len()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(train_df["char_len"].dropna(), bins=50, color="steelblue", edgecolor="white")
axes[0].set_title("Character length distribution (AfriSenti train)")
axes[0].set_xlabel("Characters")
axes[0].set_ylabel("Count")

axes[1].hist(train_df["word_len"].dropna(), bins=40, color="coral", edgecolor="white")
axes[1].set_title("Word count distribution (AfriSenti train)")
axes[1].set_xlabel("Words")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig(FIGURES / "tweet_length_distribution.png", dpi=150)
plt.close()
print("Saved tweet_length_distribution.png")
print(train_df[["char_len", "word_len"]].describe().to_string())

# ─── 5. Token frequency (after preprocessing) ─────────────────────────────────
print("\nApplying preprocessing to compute vocab …")
train_df["tokens"] = train_df["tweet"].apply(preprocess_text)
all_tokens = [tok for toks in train_df["tokens"] for tok in toks]
freq = Counter(all_tokens)
print(f"Vocabulary size (AfriSenti train): {len(freq):,}")
print("Top-20 tokens:", freq.most_common(20))

top_n = 30
top_tokens, top_counts = zip(*freq.most_common(top_n))
fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(range(top_n), top_counts, color="mediumpurple")
ax.set_xticks(range(top_n))
ax.set_xticklabels(top_tokens, rotation=45, ha="right", fontsize=9)
ax.set_title(f"Top {top_n} most frequent Amharic tokens (after stopword removal)")
ax.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig(FIGURES / "top_tokens.png", dpi=150)
plt.close()
print("Saved top_tokens.png")

# ─── 6. Per-class token frequency ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, lbl in zip(axes, ["positive", "negative", "neutral"]):
    subset = train_df[train_df["label"] == lbl]
    tokens = [t for toks in subset["tokens"] for t in toks]
    c = Counter(tokens)
    if not c:
        continue
    top_t, top_c = zip(*c.most_common(20))
    ax.bar(range(20), top_c, color={"positive": "#4CAF50", "negative": "#F44336", "neutral": "#2196F3"}[lbl])
    ax.set_xticks(range(20))
    ax.set_xticklabels(top_t, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"Top-20 tokens — {lbl} ({len(subset):,} tweets)")
    ax.set_ylabel("Frequency")
plt.suptitle("Per-class top tokens (AfriSenti train)", fontsize=13)
plt.tight_layout()
plt.savefig(FIGURES / "per_class_top_tokens.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved per_class_top_tokens.png")

# ─── 7. Train/val/test split distribution ─────────────────────────────────────
splits_data = {}
for name in ("train", "val", "test"):
    df = pd.read_csv(DATA_PROC / f"{name}.csv")
    splits_data[name] = df["label"].map(label_map).value_counts().reindex(
        ["positive", "negative", "neutral"], fill_value=0
    )

x = np.arange(3)
width = 0.25
labels_order = ["positive", "negative", "neutral"]
colors = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}

fig, ax = plt.subplots(figsize=(9, 5))
for i, (split, counts) in enumerate(splits_data.items()):
    ax.bar(x + i * width, [counts[l] for l in labels_order], width, label=split,
           color=colors[split])
ax.set_xticks(x + width)
ax.set_xticklabels(labels_order)
ax.set_title("Label distribution across train / val / test splits")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES / "split_distribution.png", dpi=150)
plt.close()
print("Saved split_distribution.png")

# ─── 8. Data quality summary ──────────────────────────────────────────────────
print("\n=== Data quality (AfriSenti train) ===")
print("Missing values:\n", train_df[["tweet", "label"]].isnull().sum().to_string())
dups = train_df.duplicated(subset=["tweet"]).sum()
print(f"Exact duplicates: {dups:,}")
empty = (train_df["tokens"].map(len) == 0).sum()
print(f"Empty after preprocessing: {empty:,}")

print(f"\nAll figures saved to {FIGURES}/")
