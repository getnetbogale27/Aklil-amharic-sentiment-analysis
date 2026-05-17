"""
Full preprocessing pipeline for Amharic sentiment analysis.

Sources:
  - data/raw/amh/train.tsv, dev.tsv, test.tsv  (AfriSenti)
  - data/raw/dataset.xlsx                        (additionally collected policy tweets)

Outputs (data/processed/):
  combined_dataset.csv   — merged raw data (tweet, label_str, label, source)
  cleaned_dataset.csv    — after preprocessing (adds tokens, cleaned_text)
  train.csv / val.csv / test.csv  — 70/15/15 stratified splits
  vocab.json             — token→index mapping
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.preprocessing.preprocess import preprocess_dataframe, preprocess_text

# ─── Paths ────────────────────────────────────────────────────────────────────
RAW_AMH = Path("data/raw/amh")
RAW_XLSX = Path("data/raw/dataset.xlsx")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Label mapping ────────────────────────────────────────────────────────────
LABEL_MAP = {"positive": 0, "negative": 1, "neutral": 2}
LABEL_STR = {0: "positive", 1: "negative", 2: "neutral"}

# ─── 1. Load AfriSenti AMH splits ─────────────────────────────────────────────
def load_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=["tweet", "label_str"],
                     encoding="utf-8")
    # Drop any accidental header row (label == "label")
    df = df[df["label_str"] != "label"].reset_index(drop=True)
    df["label_str"] = df["label_str"].str.strip().str.lower()
    df = df[df["label_str"].isin(LABEL_MAP)].reset_index(drop=True)
    df["label"] = df["label_str"].map(LABEL_MAP)
    df["source"] = "afrisenti"
    return df


splits = {}
for split in ("train", "dev", "test"):
    p = RAW_AMH / f"{split}.tsv"
    splits[split] = load_tsv(p)
    print(f"  AfriSenti {split}: {len(splits[split]):,} rows")

afrisenti_df = pd.concat(splits.values(), ignore_index=True)
print(f"  AfriSenti total: {len(afrisenti_df):,}")

# ─── 2. Load dataset.xlsx ─────────────────────────────────────────────────────
xlsx_df = pd.read_excel(RAW_XLSX)
xlsx_df = xlsx_df.rename(columns={"Text": "tweet", "Sentiment": "label_str"})
xlsx_df["label_str"] = xlsx_df["label_str"].str.strip().str.lower()
xlsx_df = xlsx_df[xlsx_df["label_str"].isin(LABEL_MAP)].reset_index(drop=True)
xlsx_df["label"] = xlsx_df["label_str"].map(LABEL_MAP)
xlsx_df["source"] = "policy"
xlsx_df = xlsx_df[["tweet", "label_str", "label", "source"]]
print(f"\n  dataset.xlsx: {len(xlsx_df):,} rows")

# ─── 3. Merge ─────────────────────────────────────────────────────────────────
combined = pd.concat(
    [afrisenti_df[["tweet", "label_str", "label", "source"]], xlsx_df],
    ignore_index=True,
)
combined = combined.dropna(subset=["tweet"]).reset_index(drop=True)
combined["tweet"] = combined["tweet"].astype(str)
print(f"\n  Combined total: {len(combined):,} rows")

combined.to_csv(OUT_DIR / "combined_dataset.csv", index=False, encoding="utf-8")
print(f"  Saved → {OUT_DIR}/combined_dataset.csv")

# ─── 4. Preprocess ────────────────────────────────────────────────────────────
print("\nRunning preprocessing …")
cleaned = preprocess_dataframe(combined, text_col="tweet", label_col="label")
# Carry over label_str and source
cleaned["label_str"] = cleaned["label"].map(LABEL_STR)
cleaned["source"] = combined.loc[cleaned.index, "source"].values
cleaned = cleaned.reset_index(drop=True)
print(f"  Rows after cleaning (non-empty): {len(cleaned):,}")

cleaned.to_csv(OUT_DIR / "cleaned_dataset.csv", index=False, encoding="utf-8")
print(f"  Saved → {OUT_DIR}/cleaned_dataset.csv")

# ─── 5. Remove exact duplicates on cleaned_text ───────────────────────────────
before_dedup = len(cleaned)
cleaned = cleaned.drop_duplicates(subset=["cleaned_text"]).reset_index(drop=True)
print(f"  After dedup: {len(cleaned):,} (removed {before_dedup - len(cleaned):,} duplicates)")

# ─── 6. Train / val / test split (70 / 15 / 15, stratified) ──────────────────
train_df, temp_df = train_test_split(
    cleaned, test_size=0.30, stratify=cleaned["label"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
)

for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    df = df.reset_index(drop=True)
    df.to_csv(OUT_DIR / f"{name}.csv", index=False, encoding="utf-8")
    print(f"  {name}: {len(df):,} rows  → {OUT_DIR}/{name}.csv")

# ─── 7. Build vocab.json ──────────────────────────────────────────────────────
all_tokens = [tok for toks in train_df["tokens"] for tok in toks]
freq = Counter(all_tokens)
# Special tokens first, then sorted by frequency
vocab = {"<PAD>": 0, "<UNK>": 1}
for tok, _ in freq.most_common():
    if tok not in vocab:
        vocab[tok] = len(vocab)

with open(OUT_DIR / "vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)
print(f"\n  Vocab size: {len(vocab):,}  → {OUT_DIR}/vocab.json")

# ─── 8. Summary statistics ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print(f"\nAfriSenti AMH samples loaded: {len(afrisenti_df):,}")
print(f"  train.tsv : {len(splits['train']):,}")
print(f"  dev.tsv   : {len(splits['dev']):,}")
print(f"  test.tsv  : {len(splits['test']):,}")

print(f"\ndataset.xlsx samples loaded : {len(xlsx_df):,}")
print(f"Combined dataset size       : {len(combined):,}")
print(f"After cleaning + dedup      : {len(cleaned):,}")

print("\nLabel distribution (cleaned dataset):")
total = len(cleaned)
for lbl_int, lbl_str in LABEL_STR.items():
    n = (cleaned["label"] == lbl_int).sum()
    print(f"  {lbl_str:10s} ({lbl_int}) : {n:,}  ({100*n/total:.1f}%)")

print("\nTrain / Val / Test split sizes:")
print(f"  train : {len(train_df):,}")
print(f"  val   : {len(val_df):,}")
print(f"  test  : {len(test_df):,}")

avg_char = cleaned["cleaned_text"].str.len().mean()
print(f"\nAverage cleaned text length : {avg_char:.1f} characters")
print(f"Vocabulary size (train)     : {len(vocab):,} tokens")

print("\nSample of 3 cleaned tweets per class:")
for lbl_int, lbl_str in LABEL_STR.items():
    subset = cleaned[cleaned["label"] == lbl_int]["cleaned_text"]
    samples = subset[subset.str.strip() != ""].head(3).tolist()
    print(f"\n  [{lbl_str.upper()}]")
    for i, s in enumerate(samples, 1):
        print(f"    {i}. {s}")

print("\n" + "=" * 60)
print("All output files saved to data/processed/")
print("=" * 60)
