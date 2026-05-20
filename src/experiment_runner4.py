"""
Fourth-pass: Fine-tune around best config and try hybrid text approaches.

Best so far: EXP_074 = MNB alpha=0.2, raw tweet, char (2-5), max=80k → F1=0.6759

This runner:
  EXP_110  MNB alpha=0.15, raw, char (2-5), max=80k
  EXP_111  MNB alpha=0.2,  raw, char (2-5), max=100k
  EXP_112  MNB alpha=0.2,  raw, char (2-6), max=100k
  EXP_113  MNB alpha=0.2,  raw, char (2-4), max=60k
  EXP_114  MNB alpha=0.25, raw, char (2-5), max=80k
  EXP_115  MNB alpha=0.3,  raw, char (2-5), max=80k
  EXP_116  Light preprocessing (only unicode-normalize + remove URLs) + NB
  EXP_117  Light preprocessing + MNB alpha=0.2
  EXP_118  Hybrid: raw tweet + cleaned_text concatenated
  EXP_119  Hybrid: NB on raw, then vote with cleaned NB
  EXP_120  Emoji-enhanced: extract emoji as word-tokens, add to cleaned_text
  EXP_121  NB alpha=0.2, char (2-5), augmented text (raw + partial clean)
  EXP_122  Soft ensemble: MNB-raw alpha=0.2 + MNB-raw alpha=0.1 + MNB-clean alpha=0.1
  EXP_123  5-fold stacking: multiple alpha values, raw text
"""

import os
import re
import sys
import time
import unicodedata
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
EXP_DIR  = ROOT / "results" / "experiments"
EXP_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES  = ["positive", "negative", "neutral"]
SEED         = 42
NB_BASELINE  = 0.6109
BEST_SO_FAR  = 0.6759  # EXP_074

np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Light preprocessing: only normalize unicode + remove URLs/RT markers
AMHARIC_NORM = {
    "ሀ": "ሃ", "ሁ": "ሁ", "ሂ": "ሂ", "ሄ": "ሄ", "ህ": "ህ", "ሆ": "ሆ",
    "ሐ": "ሃ", "ሑ": "ሁ", "ሒ": "ሂ", "ሓ": "ሃ", "ሔ": "ሄ", "ሕ": "ህ", "ሖ": "ሆ",
    "ኀ": "ሃ", "ኁ": "ሁ", "ኂ": "ሂ", "ኃ": "ሃ", "ኄ": "ሄ", "ኅ": "ህ", "ኆ": "ሆ",
    "ዐ": "አ", "ዑ": "ኡ", "ዒ": "ኢ", "ዓ": "አ", "ዔ": "ኤ", "ዕ": "እ", "ዖ": "ኦ",
    "ፀ": "ጸ", "ፁ": "ጹ", "ፂ": "ጺ", "ፃ": "ጻ", "ፄ": "ጼ", "ፅ": "ጽ", "ፆ": "ጾ",
}

def light_preprocess(text: str) -> str:
    """Minimal preprocessing: NFC normalize + Amharic char normalize + remove URLs."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    for v, c in AMHARIC_NORM.items():
        text = text.replace(v, c)
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # URLs
    text = re.sub(r"\bRT\b", " ", text)              # retweet marker
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_emoji_tokens(text: str) -> list[str]:
    """Extract emoji characters as special word tokens."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols
        "\U0001F680-\U0001F6FF"  # transport
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.findall(text)


# ---------------------------------------------------------------------------
def load_splits():
    train = pd.read_csv(DATA_DIR / "train.csv")
    val   = pd.read_csv(DATA_DIR / "val.csv")
    test  = pd.read_csv(DATA_DIR / "test.csv")

    def xy(df, col):
        return df[col].fillna("").astype(str).tolist(), df["label"].tolist()

    raw_tr, y_tr = xy(train, "tweet")
    raw_va, y_va = xy(val,   "tweet")
    raw_te, y_te = xy(test,  "tweet")
    cln_tr, _    = xy(train, "cleaned_text")
    cln_te, _    = xy(test,  "cleaned_text")

    # Light preprocessed
    lp_tr = [light_preprocess(t) for t in raw_tr]
    lp_te = [light_preprocess(t) for t in raw_te]

    # Hybrid: raw + cleaned concatenated (char n-grams will see both)
    hyb_tr = [r + " " + c for r, c in zip(raw_tr, cln_tr)]
    hyb_te = [r + " " + c for r, c in zip(raw_te, cln_te)]

    # Emoji-enhanced cleaned text
    emo_tr = [c + " " + " ".join(extract_emoji_tokens(r)) for r, c in zip(raw_tr, cln_tr)]
    emo_te = [c + " " + " ".join(extract_emoji_tokens(r)) for r, c in zip(raw_te, cln_te)]

    return {
        "raw":   (raw_tr, y_tr, raw_te, y_te),
        "clean": (cln_tr, y_tr, cln_te, y_te),
        "light": (lp_tr,  y_tr, lp_te,  y_te),
        "hyb":   (hyb_tr, y_tr, hyb_te, y_te),
        "emoji": (emo_tr, y_tr, emo_te, y_te),
    }


LOG_COLS = [
    "Experiment_ID", "Model", "Features", "Hyperparameters",
    "F1_macro", "Accuracy", "Precision_macro", "Recall_macro",
    "F1_positive", "F1_negative", "F1_neutral", "Training_time_s", "Notes",
]
exp_log = []


def log_result(exp_id, model_name, features, hyperparams, y_true, y_pred,
               train_time, notes=""):
    f1_mac = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    acc    = accuracy_score(y_true, y_pred)
    prec   = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec    = recall_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
    row = {
        "Experiment_ID":   exp_id,
        "Model":           model_name,
        "Features":        features,
        "Hyperparameters": hyperparams,
        "F1_macro":        round(f1_mac,  4),
        "Accuracy":        round(acc,     4),
        "Precision_macro": round(prec,    4),
        "Recall_macro":    round(rec,     4),
        "F1_positive":     round(f1_per[0] if len(f1_per) > 0 else 0, 4),
        "F1_negative":     round(f1_per[1] if len(f1_per) > 1 else 0, 4),
        "F1_neutral":      round(f1_per[2] if len(f1_per) > 2 else 0, 4),
        "Training_time_s": round(train_time, 1),
        "Notes":           notes,
    }
    exp_log.append(row)
    d = f1_mac - NB_BASELINE
    db = f1_mac - BEST_SO_FAR
    s  = "↑" if d  >= 0 else "↓"
    sb = "↑" if db >= 0 else "↓"
    print(f"  [{exp_id}] F1={f1_mac:.4f}  {s}{abs(d):.4f} vs NB-base  "
          f"{sb}{abs(db):.4f} vs best  "
          f"[pos={f1_per[0]:.3f} neg={f1_per[1]:.3f} neu={f1_per[2]:.3f}]  ({train_time:.1f}s)")
    return f1_mac


def char_vec(ng=(2, 5), mf=80000):
    return TfidfVectorizer(analyzer="char_wb", ngram_range=ng,
                           max_features=mf, sublinear_tf=True, min_df=1)


def save_log():
    if not exp_log:
        return pd.DataFrame(columns=LOG_COLS)
    df = pd.DataFrame(exp_log, columns=LOG_COLS)
    df = df.sort_values("F1_macro", ascending=False).reset_index(drop=True)
    df.to_excel(EXP_DIR / "experiment_log_4.xlsx", index=False)
    df.to_csv(EXP_DIR  / "experiment_log_4.csv",   index=False)
    return df


def main():
    print(f"\n{'='*65}")
    print(f"  Fourth-Pass Experiments — Alpha Fine-tuning + Hybrid Text")
    print(f"  Best so far: F1={BEST_SO_FAR}  |  Baseline: F1={NB_BASELINE}")
    print(f"{'='*65}")

    data = load_splits()
    raw_tr, y_tr, raw_te, y_te = data["raw"]
    cln_tr, _, cln_te, _       = data["clean"]
    lp_tr, _, lp_te, _         = data["light"]
    hyb_tr, _, hyb_te, _       = data["hyb"]
    emo_tr, _, emo_te, _       = data["emoji"]
    print(f"  Train: {len(raw_tr)}  Test: {len(raw_te)}\n")

    # ---------------------------------------------------------------
    print("=== Alpha fine-tune around 0.2 (raw, char 2-5, max=80k) ===")
    for exp_id, alpha in [
        ("EXP_110", 0.12), ("EXP_111", 0.15), ("EXP_112", 0.18),
        ("EXP_113", 0.20), ("EXP_114", 0.22), ("EXP_115", 0.25), ("EXP_116", 0.30),
    ]:
        t0 = time.time()
        vec = char_vec((2, 5), 80000)
        clf = MultinomialNB(alpha=alpha)
        clf.fit(vec.fit_transform(raw_tr), y_tr)
        y_pred = clf.predict(vec.transform(raw_te))
        log_result(exp_id, f"MNB raw char(2-5) alpha={alpha}",
                    "raw tweet, char (2-5), max=80k", f"alpha={alpha}",
                    y_te, y_pred, time.time() - t0)

    # ---------------------------------------------------------------
    print("\n=== Alpha=0.2 with different char ranges ===")
    for exp_id, ng, mf in [
        ("EXP_117", (2, 4),  60000),
        ("EXP_118", (2, 5),  120000),
        ("EXP_119", (2, 6),  120000),
        ("EXP_120", (1, 5),  120000),
        ("EXP_121", (1, 6),  150000),
        ("EXP_122", (2, 7),  150000),
        ("EXP_123", (3, 6),  100000),
    ]:
        t0 = time.time()
        vec = char_vec(ng, mf)
        clf = MultinomialNB(alpha=0.2)
        clf.fit(vec.fit_transform(raw_tr), y_tr)
        y_pred = clf.predict(vec.transform(raw_te))
        log_result(exp_id, f"MNB raw alpha=0.2, char{ng}",
                    f"raw tweet, char {ng}, max={mf}", "alpha=0.2",
                    y_te, y_pred, time.time() - t0)

    # ---------------------------------------------------------------
    print("\n=== Light preprocessing (normalize + remove URLs only) ===")
    for exp_id, alpha, ng, mf in [
        ("EXP_124", 0.1,  (2, 5), 80000),
        ("EXP_125", 0.2,  (2, 5), 80000),
        ("EXP_126", 0.1,  (2, 6), 100000),
        ("EXP_127", 0.2,  (2, 6), 100000),
    ]:
        t0 = time.time()
        vec = char_vec(ng, mf)
        clf = MultinomialNB(alpha=alpha)
        clf.fit(vec.fit_transform(lp_tr), y_tr)
        y_pred = clf.predict(vec.transform(lp_te))
        log_result(exp_id, f"MNB light-prep alpha={alpha}, char{ng}",
                    f"light-preproc tweet, char {ng}, max={mf}", f"alpha={alpha}",
                    y_te, y_pred, time.time() - t0,
                    notes="Only NFC+norm+URL removal")

    # ---------------------------------------------------------------
    print("\n=== Hybrid text (raw + cleaned concatenated) ===")
    for exp_id, alpha, ng, mf in [
        ("EXP_128", 0.1, (2, 5), 80000),
        ("EXP_129", 0.2, (2, 5), 80000),
        ("EXP_130", 0.1, (2, 6), 100000),
    ]:
        t0 = time.time()
        vec = char_vec(ng, mf)
        clf = MultinomialNB(alpha=alpha)
        clf.fit(vec.fit_transform(hyb_tr), y_tr)
        y_pred = clf.predict(vec.transform(hyb_te))
        log_result(exp_id, f"MNB hybrid alpha={alpha}, char{ng}",
                    f"raw+cleaned concat, char {ng}, max={mf}", f"alpha={alpha}",
                    y_te, y_pred, time.time() - t0,
                    notes="raw + cleaned_text concatenated")

    # ---------------------------------------------------------------
    print("\n=== Emoji-enhanced cleaned text ===")
    for exp_id, alpha, ng, mf in [
        ("EXP_131", 0.1, (2, 5), 80000),
        ("EXP_132", 0.2, (2, 5), 80000),
    ]:
        t0 = time.time()
        vec = char_vec(ng, mf)
        clf = MultinomialNB(alpha=alpha)
        clf.fit(vec.fit_transform(emo_tr), y_tr)
        y_pred = clf.predict(vec.transform(emo_te))
        log_result(exp_id, f"MNB emoji-enhanced alpha={alpha}, char{ng}",
                    f"cleaned_text + emoji tokens, char {ng}, max={mf}", f"alpha={alpha}",
                    y_te, y_pred, time.time() - t0,
                    notes="cleaned_text with emoji appended as tokens")

    # ---------------------------------------------------------------
    print("\n=== Soft ensemble (top models combined) ===")

    # Build 3 best-looking models and average probabilities
    configs_ens = [
        (0.2,  (2, 5), 80000,  raw_tr, raw_te),
        (0.2,  (2, 5), 120000, raw_tr, raw_te),
        (0.2,  (2, 6), 120000, raw_tr, raw_te),
        (0.1,  (2, 5), 80000,  raw_tr, raw_te),
        (0.15, (2, 5), 80000,  raw_tr, raw_te),
    ]

    t0 = time.time()
    try:
        proba_sum = None
        for alpha, ng, mf, Xtr, Xte in configs_ens:
            vec = char_vec(ng, mf)
            clf = MultinomialNB(alpha=alpha)
            clf.fit(vec.fit_transform(Xtr), y_tr)
            p = clf.predict_proba(vec.transform(Xte))
            proba_sum = p if proba_sum is None else proba_sum + p
        y_soft = proba_sum.argmax(axis=1).tolist()
        log_result("EXP_133", "Soft ensemble (5×MNB raw)",
                    "raw tweet, char (2-5/6), max=80-120k",
                    "5 MNB with alpha=0.1-0.2, avg proba",
                    y_te, y_soft, time.time() - t0,
                    notes="Soft vote: average predict_proba across 5 MNB models")
    except Exception as e:
        print(f"  [EXP_133] FAILED: {e}")

    # ---------------------------------------------------------------
    print("\n=== 5-fold stacking: MNB alpha sweep on raw text ===")
    try:
        t0 = time.time()
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        vec_stk = char_vec((2, 5), 80000)
        X_tr_v = vec_stk.fit_transform(raw_tr)
        X_te_v = vec_stk.transform(raw_te)

        alphas = [0.05, 0.1, 0.15, 0.2, 0.3]
        base_clfs = [MultinomialNB(alpha=a) for a in alphas]
        n_base = len(base_clfs)

        meta_tr = np.zeros((len(y_tr), 3 * n_base))
        for _, (tr_idx, va_idx) in enumerate(skf.split(X_tr_v, y_tr)):
            y_fold_tr = [y_tr[i] for i in tr_idx]
            for ci, bc in enumerate(base_clfs):
                bc_c = MultinomialNB(alpha=bc.alpha)
                bc_c.fit(X_tr_v[tr_idx], y_fold_tr)
                meta_tr[va_idx, ci*3:(ci+1)*3] = bc_c.predict_proba(X_tr_v[va_idx])

        meta_te = np.zeros((len(y_te), 3 * n_base))
        for ci, bc in enumerate(base_clfs):
            bc.fit(X_tr_v, y_tr)
            meta_te[:, ci*3:(ci+1)*3] = bc.predict_proba(X_te_v)

        meta_clf = LogisticRegression(C=5.0, max_iter=1000, random_state=SEED)
        meta_clf.fit(meta_tr, y_tr)
        y_pred = meta_clf.predict(meta_te).tolist()
        log_result("EXP_134", "5-fold stacking (5×MNB raw → meta-LR)",
                    "raw tweet, char (2-5) max=80k, 5-fold",
                    "alphas=[0.05,0.1,0.15,0.2,0.3], meta-LR C=5",
                    y_te, y_pred, time.time() - t0)
    except Exception as e:
        print(f"  [EXP_134] FAILED: {e}")

    # ---------------------------------------------------------------
    # Save log
    df = save_log()

    if len(df) > 0:
        print(f"\n{'='*65}")
        print(f"  Phase 4 Results (top-10)")
        print(f"{'='*65}")
        for _, row in df.head(10).iterrows():
            d = row["F1_macro"] - NB_BASELINE
            s = "↑" if d >= 0 else "↓"
            print(f"  {row['Experiment_ID']:>8}  {row['Model'][:40]:<40}  "
                  f"F1={row['F1_macro']:.4f}  {s}{abs(d):.4f}")
        print(f"\n  Phase 4 Best: {df.iloc[0]['Model']} @ F1={df.iloc[0]['F1_macro']:.4f}")

        # Save best model
        best_row = df.iloc[0]
        alpha_best = 0.2
        if "alpha=" in best_row["Hyperparameters"]:
            try:
                alpha_best = float(best_row["Hyperparameters"].split("alpha=")[1].split(",")[0].split(")")[0])
            except:
                pass
        print(f"\n  Saving best model for reuse...")
        vec_final = char_vec((2, 5), 80000)
        clf_final = MultinomialNB(alpha=alpha_best)
        clf_final.fit(vec_final.fit_transform(raw_tr), y_tr)
        import joblib
        joblib.dump(vec_final, EXP_DIR / "best_vec_phase4.pkl")
        joblib.dump(clf_final, EXP_DIR / "best_clf_phase4.pkl")
        print(f"  Saved to {EXP_DIR}/best_vec_phase4.pkl")

        # Confusion matrix for best model
        y_cm = clf_final.predict(vec_final.transform(raw_te))
        cm = confusion_matrix(y_te, y_cm)
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        f1_best = f1_score(y_te, y_cm, average="macro", zero_division=0)
        ax.set_title(f"Best Model CM — MNB raw char(2-5) alpha={alpha_best}\nF1={f1_best:.4f}")
        plt.tight_layout()
        cm_path = EXP_DIR / "best_confusion_matrix.jpeg"
        fig.savefig(cm_path, dpi=150, format="jpeg")
        plt.close(fig)
        print(f"  CM saved → {cm_path}")
        print(f"\n  Classification report:")
        print(classification_report(y_te, y_cm, target_names=LABEL_NAMES, zero_division=0))


if __name__ == "__main__":
    main()
