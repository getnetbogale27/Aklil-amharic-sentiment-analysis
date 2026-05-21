"""
Generate final transformer comparison report.
Collects all transformer run results and produces:
  - results/transformer_runs/transformer_comparison.csv
  - results/transformer_runs/all_transformers_comparison.jpeg
  - Console: final leaderboard
"""

import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("results/transformer_runs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Collect transformer run results ───────────────────────────────────────────
records = []
for results_path in sorted(glob.glob(str(OUTPUT_DIR / "**/results.json"), recursive=True)):
    with open(results_path) as f:
        r = json.load(f)
    tr = r.get("test_results", {})
    if not tr:
        continue
    if "ensemble_type" in r:
        # Add ensemble as its own row
        records.append({
            "Model":        f"Ensemble (MNB+Transformer α={r.get('best_alpha',0.9)})",
            "Type":         "Ensemble",
            "Parameters":   0,
            "F1_macro":     tr.get("f1_macro", 0),
            "Accuracy":     tr.get("accuracy", 0),
            "Precision":    tr.get("precision_macro", 0),
            "Recall":       tr.get("recall_macro", 0),
            "F1_weighted":  tr.get("f1_weighted", 0),
            "Train_time_min": 0,
            "Epochs":       0,
            "Run_dir":      str(Path(results_path).parent),
        })
        continue
    records.append({
        "Model":        r.get("experiment_name", Path(results_path).parent.name),
        "Type":         "Offline (random init)" if r.get("offline_mode") else "Pretrained",
        "Parameters":   r.get("parameters", 0),
        "F1_macro":     tr.get("f1_macro", 0),
        "Accuracy":     tr.get("accuracy", 0),
        "Precision":    tr.get("precision_macro", 0),
        "Recall":       tr.get("recall_macro", 0),
        "F1_weighted":  tr.get("f1_weighted", 0),
        "Train_time_min": r.get("train_time_minutes", 0),
        "Epochs":       r.get("epochs_trained", 0),
        "Run_dir":      str(Path(results_path).parent),
    })

# ── Add ML baselines ───────────────────────────────────────────────────────────
import joblib
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

DATA_DIR = Path("data/processed")
test_df  = pd.read_csv(DATA_DIR / "test.csv")
texts    = test_df["tweet"].fillna("").astype(str).tolist()
labels   = test_df["label"].tolist()

# Phase 4 best MNB
try:
    vec = joblib.load("results/experiments/best_vec_phase4.pkl")
    clf = joblib.load("results/experiments/best_clf_phase4.pkl")
    X   = vec.transform(texts)
    p   = clf.predict(X)
    records.append({
        "Model":        "MNB-char-best (phase 4)",
        "Type":         "ML baseline",
        "Parameters":   0,
        "F1_macro":     round(f1_score(labels, p, average="macro"), 4),
        "Accuracy":     round(accuracy_score(labels, p), 4),
        "Precision":    round(precision_score(labels, p, average="macro", zero_division=0), 4),
        "Recall":       round(recall_score(labels, p, average="macro", zero_division=0), 4),
        "F1_weighted":  round(f1_score(labels, p, average="weighted"), 4),
        "Train_time_min": 0,
        "Epochs":       0,
        "Run_dir":      "results/experiments/",
    })
except Exception as e:
    print(f"  Warning: could not load phase4 model: {e}")

# Bi-LSTM from existing metrics
records.append({
    "Model":        "Bi-LSTM (char-level)",
    "Type":         "Neural (random init)",
    "Parameters":   0,
    "F1_macro":     0.5776,
    "Accuracy":     0.5759,
    "Precision":    0.5778,
    "Recall":       0.5792,
    "F1_weighted":  0.5746,
    "Train_time_min": 0,
    "Epochs":       0,
    "Run_dir":      "",
})

# XGBoost and LightGBM baselines
boosting_path = Path("results/experiments/boosting_results.json")
if boosting_path.exists():
    import json as _json
    with open(boosting_path) as f:
        boosting = _json.load(f)
    for name, r in boosting.items():
        records.append({
            "Model":        name,
            "Type":         "ML baseline",
            "Parameters":   0,
            "F1_macro":     r.get("f1_macro", 0),
            "Accuracy":     r.get("accuracy", 0),
            "Precision":    0,
            "Recall":       0,
            "F1_weighted":  0,
            "Train_time_min": round(r.get("train_time_sec", 0) / 60, 1),
            "Epochs":       0,
            "Run_dir":      "results/experiments/",
        })

# ── Build DataFrame ────────────────────────────────────────────────────────────
df = pd.DataFrame(records)
df = df.sort_values("F1_macro", ascending=False).reset_index(drop=True)
df.to_csv(OUTPUT_DIR / "transformer_comparison.csv", index=False)
print(f"Saved: {OUTPUT_DIR / 'transformer_comparison.csv'}")

# ── Print leaderboard ──────────────────────────────────────────────────────────
MNB_BEST  = df[df["Type"] == "ML baseline"]["F1_macro"].max() if df["Type"].eq("ML baseline").any() else 0.6759
BILSTM_F1 = 0.5776

print(f"\n{'='*72}")
print(f"  FINAL LEADERBOARD — All Models Ranked by F1_macro")
print(f"{'='*72}")
print(f"  {'Rank':<5} {'Model':<35} {'F1_macro':<10} {'vs MNB baseline'}")
print(f"  {'─'*70}")
for i, row in df.iterrows():
    diff = row["F1_macro"] - MNB_BEST
    diff_str = f"{diff:+.4f}" if MNB_BEST > 0 else "N/A"
    marker = " ← BEST" if i == 0 else ""
    print(f"  #{i+1:<4} {row['Model']:<35} {row['F1_macro']:.4f}     {diff_str}{marker}")

print(f"\n  MNB baseline (best ML): {MNB_BEST:.4f}")
print(f"  Bi-LSTM:                {BILSTM_F1:.4f}")
print(f"\n  TARGET: 0.8000")
print(f"  BEST:   {df['F1_macro'].max():.4f}")
gap = 0.8000 - df["F1_macro"].max()
print(f"  GAP:    {gap:.4f} points to 0.80 target")
print(f"\n  NOTE: Network policy blocked HuggingFace downloads. Pretrained models")
print(f"  (AfriBERTa, Afro-XLM-R) expected to achieve F1 ≈ 0.72-0.78 based on")
print(f"  literature. Scripts ready in src/train_transformer_cpu.py for when")
print(f"  network is available.")
print(f"{'='*72}\n")

# ── Bar chart ──────────────────────────────────────────────────────────────────
color_map = {
    "ML baseline":          "#2196F3",
    "Offline (random init)": "#FF9800",
    "Neural (random init)":  "#E91E63",
    "Pretrained":           "#4CAF50",
    "Ensemble":             "#9C27B0",
}
default_color = "#9E9E9E"

fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.6 + 2)))
colors  = [color_map.get(t, default_color) for t in df["Type"]]
bars    = ax.barh(df["Model"][::-1], df["F1_macro"][::-1], color=colors[::-1], height=0.6)

ax.axvline(x=MNB_BEST,  color="#2196F3", linestyle="--", linewidth=1.5,
           label=f"MNB-char-best ({MNB_BEST:.4f})")
ax.axvline(x=BILSTM_F1, color="#E91E63", linestyle=":", linewidth=1.2,
           label=f"Bi-LSTM ({BILSTM_F1:.4f})")
ax.axvline(x=0.80,       color="red",     linestyle="-", linewidth=1.2, alpha=0.4,
           label="Target (0.80)")

for bar, val in zip(bars, df["F1_macro"][::-1]):
    ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=9)

patches = [mpatches.Patch(color=v, label=k) for k, v in color_map.items()]
ax.legend(handles=patches + [
    plt.Line2D([0], [0], color="#2196F3", linestyle="--", label=f"MNB-best ({MNB_BEST:.4f})"),
    plt.Line2D([0], [0], color="red",     linestyle="-",  alpha=0.4, label="Target (0.80)"),
], loc="lower right", fontsize=9)

ax.set_xlabel("F1_macro (test set)", fontsize=12)
ax.set_title("Amharic Sentiment Analysis — All Models\n(Transformers offline: no pretrained weights available)",
             fontsize=13, fontweight="bold")
ax.set_xlim(0, 0.85)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "all_transformers_comparison.jpeg", dpi=150, format="jpeg")
plt.close()
print(f"Saved: {OUTPUT_DIR / 'all_transformers_comparison.jpeg'}")
