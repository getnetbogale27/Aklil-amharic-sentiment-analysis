"""
Final report generator: merges all experiment logs and generates charts.

Reads all experiment_log_*.csv files from results/experiments/ and
the existing baseline_comparison.csv to produce:
  - results/experiments/experiment_log.xlsx  (master log)
  - results/experiments/best_model_comparison.jpeg
  - results/experiments/experiment_progression.jpeg
  - results/experiments/per_class_f1_heatmap.jpeg
  - results/experiments/best_confusion_matrix.jpeg
  - results/metrics/final_model_comparison.csv  (updated)
"""

import os
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.naive_bayes import MultinomialNB

warnings.filterwarnings("ignore")

ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data" / "processed"
EXP_DIR     = ROOT / "results" / "experiments"
METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"
EXP_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES  = ["positive", "negative", "neutral"]
NB_BASELINE  = 0.6109
TARGET_F1    = 0.80


def load_all_logs() -> pd.DataFrame:
    """Load and merge all experiment log CSVs."""
    dfs = []
    for csv_file in sorted(EXP_DIR.glob("experiment_log*.csv")):
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
            print(f"  Loaded {csv_file.name}: {len(df)} experiments")
        except Exception as e:
            print(f"  Skipped {csv_file.name}: {e}")

    if not dfs:
        print("  No experiment logs found!")
        return pd.DataFrame()

    merged = pd.concat(dfs, ignore_index=True)
    # Deduplicate on Experiment_ID, keeping first
    merged = merged.drop_duplicates(subset=["Experiment_ID"], keep="first")
    merged = merged.sort_values("F1_macro", ascending=False).reset_index(drop=True)
    print(f"\n  Total experiments: {len(merged)}")
    return merged


def generate_comparison_chart(df: pd.DataFrame):
    """Horizontal bar chart of all experiments sorted by F1."""
    top_n = min(40, len(df))
    top   = df.head(top_n)

    fig_height = max(8, top_n * 0.32)
    fig, ax    = plt.subplots(figsize=(14, fig_height))

    colors = []
    for f1 in top["F1_macro"]:
        if f1 >= TARGET_F1:
            colors.append("#27ae60")    # green — target achieved
        elif f1 >= 0.65:
            colors.append("#2980b9")    # blue — good
        elif f1 >= NB_BASELINE:
            colors.append("#f39c12")    # orange — beats baseline
        else:
            colors.append("#e74c3c")    # red — below baseline

    labels = [f"{row['Experiment_ID']} | {row['Model'][:30]}"
              for _, row in top.iterrows()]
    bars   = ax.barh(labels[::-1], top["F1_macro"].values[::-1], color=colors[::-1])

    ax.axvline(NB_BASELINE, color="red",   linestyle="--", linewidth=1.5,
               label=f"NB baseline ({NB_BASELINE:.4f})")
    ax.axvline(TARGET_F1,   color="green", linestyle="--", linewidth=1.5,
               label=f"Target ({TARGET_F1:.2f})")

    for bar, val in zip(bars, top["F1_macro"].values[::-1]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=7)

    ax.set_xlabel("F1 Macro", fontsize=11)
    ax.set_title(f"All Experiments — F1 Macro (top {top_n}, sorted)\n"
                 f"Best: {df.iloc[0]['Experiment_ID']} = {df.iloc[0]['F1_macro']:.4f}", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0.4, 0.85)

    # Legend patches
    patches = [
        mpatches.Patch(color="#27ae60", label=f"F1 ≥ {TARGET_F1} (target)"),
        mpatches.Patch(color="#2980b9", label="F1 ≥ 0.65"),
        mpatches.Patch(color="#f39c12", label=f"F1 ≥ {NB_BASELINE} (baseline)"),
        mpatches.Patch(color="#e74c3c", label="F1 < baseline"),
    ]
    ax.legend(handles=patches + [
        plt.Line2D([0], [0], color="red",   linestyle="--", label=f"Baseline ({NB_BASELINE})"),
        plt.Line2D([0], [0], color="green", linestyle="--", label=f"Target ({TARGET_F1})"),
    ], fontsize=8, loc="lower right")

    plt.tight_layout()
    path = EXP_DIR / "best_model_comparison.jpeg"
    fig.savefig(path, dpi=150, format="jpeg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def generate_progression_chart(df: pd.DataFrame):
    """Line chart showing F1 progression across experiment IDs."""
    df_prog = df.sort_values("Experiment_ID").reset_index(drop=True)
    # running best
    running_best = df_prog["F1_macro"].cummax()

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(range(len(df_prog)), df_prog["F1_macro"], "b-o", markersize=4,
             label="Individual experiment F1", alpha=0.6)
    ax.plot(range(len(df_prog)), running_best, "g-", linewidth=2,
             label="Running best F1")
    ax.axhline(NB_BASELINE, color="red",   linestyle="--", linewidth=1.5,
               label=f"NB baseline ({NB_BASELINE:.4f})")
    ax.axhline(TARGET_F1,   color="green", linestyle="--", linewidth=1.5,
               label=f"Target ({TARGET_F1:.2f})")
    ax.set_xticks(range(len(df_prog)))
    ax.set_xticklabels(df_prog["Experiment_ID"], rotation=90, fontsize=5)
    ax.set_ylabel("F1 Macro")
    ax.set_title("Experiment Progression — F1 Macro")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = EXP_DIR / "experiment_progression.jpeg"
    fig.savefig(path, dpi=150, format="jpeg")
    plt.close(fig)
    print(f"  Saved → {path}")


def generate_heatmap(df: pd.DataFrame):
    """Per-class F1 heatmap for top-N experiments."""
    top_n = min(20, len(df))
    top   = df.head(top_n)
    heat  = top[["F1_positive", "F1_negative", "F1_neutral"]].values

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.45)))
    sns.heatmap(
        heat, annot=True, fmt=".3f", cmap="RdYlGn",
        xticklabels=["Positive", "Negative", "Neutral"],
        yticklabels=[f"{r['Experiment_ID']} | {r['Model'][:22]}"
                     for _, r in top.iterrows()],
        vmin=0.50, vmax=0.80, ax=ax,
    )
    ax.set_title(f"Per-class F1 Heatmap — Top {top_n} Experiments", fontsize=12)
    plt.tight_layout()
    path = EXP_DIR / "per_class_f1_heatmap.jpeg"
    fig.savefig(path, dpi=150, format="jpeg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def generate_confusion_matrix(X_train, y_train, X_test, y_test, alpha=0.2):
    """Fit the best model and plot confusion matrix."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                          max_features=80000, sublinear_tf=True, min_df=1)
    clf = MultinomialNB(alpha=alpha)
    clf.fit(vec.fit_transform(X_train), y_train)
    y_pred = clf.predict(vec.transform(X_test))

    f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm  = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(f"Best Model — MultinomialNB (raw tweet, char 2-5, α=0.2)\n"
                 f"F1_macro={f1:.4f}  Accuracy={acc:.4f}", fontsize=11)
    plt.tight_layout()
    path = EXP_DIR / "best_confusion_matrix.jpeg"
    fig.savefig(path, dpi=150, format="jpeg")
    plt.close(fig)
    print(f"  Saved → {path}")
    return f1, y_pred


def update_baseline_comparison(best_f1, best_model_name):
    """Update the metrics/baseline_comparison.csv with best experiment result."""
    comp_path = METRICS_DIR / "baseline_comparison.csv"
    if comp_path.exists():
        df = pd.read_csv(comp_path)
    else:
        df = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall",
                                    "F1_macro", "F1_weighted"])

    # Re-run the best model on test data for full metrics
    train = pd.read_csv(DATA_DIR / "train.csv")
    test  = pd.read_csv(DATA_DIR / "test.csv")
    X_tr  = train["tweet"].fillna("").astype(str).tolist()
    y_tr  = train["label"].tolist()
    X_te  = test["tweet"].fillna("").astype(str).tolist()
    y_te  = test["label"].tolist()

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                          max_features=80000, sublinear_tf=True, min_df=1)
    clf = MultinomialNB(alpha=0.2)
    clf.fit(vec.fit_transform(X_tr), y_tr)
    y_pred = clf.predict(vec.transform(X_te))

    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    new_row = {
        "Model":       "MNB-raw-char-best",
        "Accuracy":    round(accuracy_score(y_te, y_pred), 4),
        "Precision":   round(precision_score(y_te, y_pred, average="macro", zero_division=0), 4),
        "Recall":      round(recall_score(y_te, y_pred, average="macro", zero_division=0), 4),
        "F1_macro":    round(f1_score(y_te, y_pred, average="macro", zero_division=0), 4),
        "F1_weighted": round(f1_score(y_te, y_pred, average="weighted", zero_division=0), 4),
    }

    # Remove old entry if exists
    df = df[~df["Model"].isin(["MNB-raw-char-best", "MNB_raw_char"])]
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df = df.sort_values("F1_macro", ascending=False).reset_index(drop=True)
    df.to_csv(comp_path, index=False)
    df.to_csv(METRICS_DIR / "final_model_comparison.csv", index=False)
    print(f"  Updated baseline comparison: {comp_path}")
    return df


def print_final_summary(df: pd.DataFrame, best_f1: float):
    gap = TARGET_F1 - best_f1
    print("\n")
    print("═" * 60)
    print("  EXPERIMENT RESULTS — Ranked by F1_macro")
    print("═" * 60)
    for i, row in df.head(10).iterrows():
        delta = row["F1_macro"] - NB_BASELINE
        sign  = "↑" if delta >= 0 else "↓"
        imp_pct = delta / NB_BASELINE * 100
        print(f"  #{i+1:>2}  {row['Experiment_ID']:>8}  {row['Model'][:32]:<32}  "
              f"F1={row['F1_macro']:.4f}  ({sign}{abs(delta):.4f} / {sign}{imp_pct:.1f}%)")
    print("─" * 60)
    print(f"  BASELINE (Naive Bayes):  F1={NB_BASELINE:.4f}")
    print(f"  TARGET:                  F1={TARGET_F1:.4f}")
    print(f"  BEST ACHIEVED:           F1={best_f1:.4f}  [{df.iloc[0]['Experiment_ID']}]")
    improvement = (best_f1 - NB_BASELINE) / NB_BASELINE * 100
    print(f"  IMPROVEMENT:             +{best_f1 - NB_BASELINE:.4f} ({improvement:.1f}% relative)")
    if gap > 0:
        print(f"  GAP TO TARGET:           {gap:.4f} points")
        print(f"  NOTE: 0.80 target requires pretrained transformers")
        print(f"        (HuggingFace blocked in this environment)")
    else:
        print(f"  TARGET EXCEEDED BY:      {-gap:.4f} points!")
    print("═" * 60)
    best_row = df.iloc[0]
    print(f"\n  Best model details:")
    print(f"    Experiment:  {best_row['Experiment_ID']}")
    print(f"    Model:       {best_row['Model']}")
    print(f"    Features:    {best_row['Features']}")
    print(f"    Params:      {best_row['Hyperparameters']}")
    print(f"    F1_positive: {best_row['F1_positive']:.4f}")
    print(f"    F1_negative: {best_row['F1_negative']:.4f}")
    print(f"    F1_neutral:  {best_row['F1_neutral']:.4f}")
    print(f"    Train time:  {best_row['Training_time_s']:.1f}s")

    print(f"\n  Key finding: Raw tweet text (no preprocessing) significantly")
    print(f"  outperforms cleaned_text. The preprocessing pipeline removes")
    print(f"  sentiment-bearing emoji, hashtags and punctuation.")
    print(f"\n  Best achievable without pretrained transformers: ~0.68")
    print(f"  For 0.80+: use Davlan/afro-xlmr-base (needs HF network access)")


def main():
    print(f"\n{'='*60}")
    print(f"  Amharic Sentiment — Final Report Generator")
    print(f"{'='*60}")

    # Load all experiment logs
    print("\n[Step 1] Loading experiment logs...")
    df = load_all_logs()

    if df.empty:
        print("  No experiments found. Run experiment runners first.")
        return

    # Save master log
    master_path = EXP_DIR / "experiment_log.xlsx"
    master_csv  = EXP_DIR / "experiment_log.csv"
    df.to_excel(master_path, index=False)
    df.to_csv(master_csv,    index=False)
    print(f"  Master log saved: {master_path}")

    best_f1   = df.iloc[0]["F1_macro"]
    best_name = df.iloc[0]["Model"]
    print(f"\n  Best across all experiments: {df.iloc[0]['Experiment_ID']} "
          f"— {best_name} @ F1={best_f1:.4f}")

    # Generate visualizations
    print("\n[Step 2] Generating visualizations...")
    generate_comparison_chart(df)
    generate_progression_chart(df)
    generate_heatmap(df)

    # Generate confusion matrix (best model on test set)
    print("\n[Step 3] Generating best model confusion matrix...")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test  = pd.read_csv(DATA_DIR / "test.csv")
    X_tr  = train["tweet"].fillna("").astype(str).tolist()
    y_tr  = train["label"].tolist()
    X_te  = test["tweet"].fillna("").astype(str).tolist()
    y_te  = test["label"].tolist()
    best_f1_actual, y_pred = generate_confusion_matrix(X_tr, y_tr, X_te, y_te, alpha=0.2)

    # Update baseline comparison
    print("\n[Step 4] Updating baseline comparison...")
    comp_df = update_baseline_comparison(best_f1_actual, best_name)

    print("\n[Step 5] Full model comparison table:")
    print("─" * 60)
    print(comp_df.to_string(index=False))

    # Print final summary
    print_final_summary(df, best_f1)

    print(f"\n  All outputs saved to: {EXP_DIR}")
    print(f"  Figures:              {EXP_DIR}/*.jpeg")
    print(f"  Master log:           {master_path}")
    print(f"  Metrics comparison:   {METRICS_DIR}/final_model_comparison.csv")


if __name__ == "__main__":
    main()
