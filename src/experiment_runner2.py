"""
Second-pass experiments focusing on insights from Phase 1.

Key finding: NB with char TF-IDF (2-5) = 0.6421 beats everything.

This file tries:
  EXP_030  ComplementNB + char TF-IDF (2-5)         [better NB variant]
  EXP_031  ComplementNB + char TF-IDF (2-6)
  EXP_032  ComplementNB + char TF-IDF (3-6)
  EXP_033  ComplementNB + char TF-IDF (2-5), alpha sweep
  EXP_034  NB + raw tweet text (no preprocessing)    [preserve more signal]
  EXP_035  ComplementNB + raw tweet text
  EXP_036  LR char only (no balanced) — test if balanced hurts
  EXP_037  SVM char only (no balanced)
  EXP_038  LR char + word, no balanced, C=5
  EXP_039  NB char, max_features=100k
  EXP_040  ComplementNB char, max_features=100k
  EXP_041  NB combined (word+char)
  EXP_042  ComplementNB combined (word+char)
  EXP_043  Stacking: NB+LR+SVM meta-LR              [stacking ensemble]
  EXP_044  Stacking: top-3 char models
  EXP_045  LR solver=saga, l1 penalty, char TF-IDF
"""

import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
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
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight
import joblib

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
EXP_DIR  = ROOT / "results" / "experiments"
EXP_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES  = ["positive", "negative", "neutral"]
SEED         = 42
NB_BASELINE  = 0.6109
PHASE1_BEST  = 0.6421  # EXP_002 char NB

np.random.seed(SEED)


# ---------------------------------------------------------------------------
def load_splits():
    train = pd.read_csv(DATA_DIR / "train.csv")
    val   = pd.read_csv(DATA_DIR / "val.csv")
    test  = pd.read_csv(DATA_DIR / "test.csv")

    def xy(df, col="cleaned_text"):
        return df[col].fillna("").astype(str).tolist(), df["label"].tolist()

    return (xy(train), xy(train, "tweet"),
            xy(val),   xy(val,   "tweet"),
            xy(test),  xy(test,  "tweet"))


# ---------------------------------------------------------------------------
LOG_COLS = [
    "Experiment_ID", "Model", "Features", "Hyperparameters",
    "F1_macro", "Accuracy", "Precision_macro", "Recall_macro",
    "F1_positive", "F1_negative", "F1_neutral",
    "Training_time_s", "Notes",
]
exp_log: list[dict] = []


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
    d1 = f1_mac - NB_BASELINE
    d2 = f1_mac - PHASE1_BEST
    s1 = "↑" if d1 >= 0 else "↓"
    s2 = "↑" if d2 >= 0 else "↓"
    print(f"  [{exp_id}] F1={f1_mac:.4f}  "
          f"{s1}{abs(d1):.4f} vs NB-base  {s2}{abs(d2):.4f} vs Phase1-best  "
          f"[pos={f1_per[0]:.3f} neg={f1_per[1]:.3f} neu={f1_per[2]:.3f}]  ({train_time:.1f}s)")
    return f1_mac


def save_log(suffix="2"):
    if not exp_log:
        print("  [skip] No experiments logged.")
        return pd.DataFrame(columns=LOG_COLS)
    df = pd.DataFrame(exp_log, columns=LOG_COLS)
    df = df.sort_values("F1_macro", ascending=False).reset_index(drop=True)
    df.to_excel(EXP_DIR / f"experiment_log_{suffix}.xlsx", index=False)
    df.to_csv(EXP_DIR  / f"experiment_log_{suffix}.csv",   index=False)
    return df


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------
def char_vec(ngram=(2, 5), max_features=50000, min_df=1):
    return TfidfVectorizer(analyzer="char_wb", ngram_range=ngram,
                           max_features=max_features, sublinear_tf=True, min_df=min_df)

def word_vec(ngram=(1, 2), max_features=20000):
    return TfidfVectorizer(analyzer="word", ngram_range=ngram,
                           max_features=max_features, sublinear_tf=True, min_df=2)

def combined_vec(wn=(1, 2), cn=(2, 5), wf=20000, cf=30000):
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word",    ngram_range=wn, max_features=wf, sublinear_tf=True, min_df=2)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=cn, max_features=cf, sublinear_tf=True, min_df=1)),
    ])


# ---------------------------------------------------------------------------
def main():
    print(f"\n{'='*65}")
    print(f"  Second-Pass Experiments — Char NB Focus")
    print(f"  Phase 1 best: {PHASE1_BEST} (char NB)  |  Baseline: {NB_BASELINE}")
    print(f"{'='*65}")

    out = load_splits()
    (X_train, y_train), (X_train_raw, _), \
    (X_val,   y_val),   (X_val_raw,   _), \
    (X_test,  y_test),  (X_test_raw,  _) = out

    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    cw = compute_class_weight("balanced", classes=np.array([0,1,2]), y=np.array(y_train))
    print(f"  Class weights: pos={cw[0]:.3f} neg={cw[1]:.3f} neu={cw[2]:.3f}\n")

    # -----------------------------------------------------------------
    print("=== ComplementNB + char TF-IDF ===")

    for exp_id, ng, alpha, max_f in [
        ("EXP_030", (2, 5), 0.1,  50000),
        ("EXP_031", (2, 6), 0.1,  60000),
        ("EXP_032", (3, 6), 0.1,  60000),
        ("EXP_033", (2, 5), 0.01, 50000),
        ("EXP_034", (2, 5), 0.5,  50000),
        ("EXP_035", (2, 5), 1.0,  50000),
        ("EXP_036", (2, 5), 0.05, 50000),
        ("EXP_037", (2, 4), 0.1,  40000),
        ("EXP_038", (3, 5), 0.1,  50000),
    ]:
        t0 = time.time()
        try:
            vec = char_vec(ng, max_f, min_df=1)
            clf = ComplementNB(alpha=alpha)
            clf.fit(vec.fit_transform(X_train), y_train)
            y_pred = clf.predict(vec.transform(X_test))
            log_result(exp_id, "ComplementNB",
                        f"char TF-IDF {ng}, max={max_f}",
                        f"alpha={alpha}",
                        y_test, y_pred, time.time() - t0)
        except Exception as e:
            print(f"  [{exp_id}] FAILED: {e}")

    # -----------------------------------------------------------------
    print("\n=== MultinomialNB alpha sweep ===")
    for exp_id, alpha, ng, max_f in [
        ("EXP_039", 0.01, (2, 5), 50000),
        ("EXP_040", 0.05, (2, 5), 50000),
        ("EXP_041", 0.5,  (2, 5), 50000),
        ("EXP_042", 1.0,  (2, 5), 50000),
        ("EXP_043", 0.1,  (2, 5), 100000),   # more features
        ("EXP_044", 0.1,  (2, 6), 100000),
    ]:
        t0 = time.time()
        try:
            vec = char_vec(ng, max_f, min_df=1)
            clf = MultinomialNB(alpha=alpha)
            clf.fit(vec.fit_transform(X_train), y_train)
            y_pred = clf.predict(vec.transform(X_test))
            log_result(exp_id, "MultinomialNB",
                        f"char TF-IDF {ng}, max={max_f}",
                        f"alpha={alpha}",
                        y_test, y_pred, time.time() - t0)
        except Exception as e:
            print(f"  [{exp_id}] FAILED: {e}")

    # -----------------------------------------------------------------
    print("\n=== NB on raw tweet text (no preprocessing) ===")
    for exp_id, clf_cls, alpha, ng, max_f in [
        ("EXP_045", ComplementNB,  0.1, (2, 5), 80000),
        ("EXP_046", MultinomialNB, 0.1, (2, 5), 80000),
        ("EXP_047", ComplementNB,  0.1, (2, 6), 100000),
    ]:
        t0 = time.time()
        try:
            vec = char_vec(ng, max_f, min_df=1)
            clf = clf_cls(alpha=alpha)
            clf.fit(vec.fit_transform(X_train_raw), y_train)
            y_pred = clf.predict(vec.transform(X_test_raw))
            name   = "ComplementNB" if clf_cls is ComplementNB else "MultinomialNB"
            log_result(exp_id, f"{name} (RAW text)",
                        f"raw tweet, char TF-IDF {ng}, max={max_f}",
                        f"alpha={alpha}",
                        y_test, y_pred, time.time() - t0,
                        notes="Uses original tweet, not cleaned_text")
        except Exception as e:
            print(f"  [{exp_id}] FAILED: {e}")

    # -----------------------------------------------------------------
    print("\n=== LR / SVM without balanced class weight ===")
    for exp_id, clf, feat_name, feat_fn in [
        ("EXP_048", LogisticRegression(C=1.0,  max_iter=2000, random_state=SEED),
         "char (2-5)", lambda: char_vec((2, 5), 50000)),
        ("EXP_049", LogisticRegression(C=5.0,  max_iter=2000, random_state=SEED),
         "char (2-5)", lambda: char_vec((2, 5), 50000)),
        ("EXP_050", LogisticRegression(C=10.0, max_iter=2000, random_state=SEED),
         "char (2-5)", lambda: char_vec((2, 5), 50000)),
        ("EXP_051", LinearSVC(C=1.0,  max_iter=5000, random_state=SEED),
         "char (2-5)", lambda: char_vec((2, 5), 50000)),
        ("EXP_052", LinearSVC(C=0.5,  max_iter=5000, random_state=SEED),
         "char (2-5)", lambda: char_vec((2, 5), 50000)),
    ]:
        t0 = time.time()
        try:
            vec = feat_fn()
            X_tr_v = vec.fit_transform(X_train)
            X_te_v = vec.transform(X_test)
            clf.fit(X_tr_v, y_train)
            y_pred = clf.predict(X_te_v)
            name   = type(clf).__name__
            params = str(clf.get_params()).replace("{", "").replace("}", "")[:60]
            log_result(exp_id, f"{name} (no balanced, {feat_name})",
                        f"char TF-IDF (2-5) only, max=50k",
                        params,
                        y_test, y_pred, time.time() - t0)
        except Exception as e:
            print(f"  [{exp_id}] FAILED: {e}")

    # -----------------------------------------------------------------
    print("\n=== NB combined (word+char) ===")
    for exp_id, clf_cls, alpha in [
        ("EXP_053", ComplementNB,  0.1),
        ("EXP_054", MultinomialNB, 0.1),
        ("EXP_055", ComplementNB,  0.01),
    ]:
        t0 = time.time()
        try:
            vec = combined_vec((1, 2), (2, 5), 20000, 30000)
            clf = clf_cls(alpha=alpha)
            clf.fit(vec.fit_transform(X_train), y_train)
            y_pred = clf.predict(vec.transform(X_test))
            name   = "ComplementNB" if clf_cls is ComplementNB else "MultinomialNB"
            log_result(exp_id, f"{name} combined (word+char)",
                        "word (1-2)+char (2-5) TF-IDF",
                        f"alpha={alpha}",
                        y_test, y_pred, time.time() - t0)
        except Exception as e:
            print(f"  [{exp_id}] FAILED: {e}")

    # -----------------------------------------------------------------
    print("\n=== Stacking ensemble ===")

    # Build meta-features: out-of-fold predictions from base models
    # Use val set predictions as meta-features for simplicity
    try:
        exp_id = "EXP_056"
        t0 = time.time()
        vec1 = char_vec((2, 5), 50000)
        vec2 = char_vec((2, 6), 60000)
        vec3 = combined_vec((1, 2), (2, 5), 20000, 30000)

        X_tr_1 = vec1.fit_transform(X_train)
        X_te_1 = vec1.transform(X_test)
        X_tr_2 = vec2.fit_transform(X_train)
        X_te_2 = vec2.transform(X_test)
        X_tr_3 = vec3.fit_transform(X_train)
        X_te_3 = vec3.transform(X_test)

        # Base models for stacking
        base_models = [
            ("nb1",  MultinomialNB(alpha=0.1)),
            ("cnb1", ComplementNB(alpha=0.1)),
            ("cnb2", ComplementNB(alpha=0.1)),
            ("lr1",  LogisticRegression(C=5.0, max_iter=2000, random_state=SEED)),
        ]
        vecs_tr  = [X_tr_1, X_tr_1, X_tr_2, X_tr_3]
        vecs_te  = [X_te_1, X_te_1, X_te_2, X_te_3]

        # Get val features for meta-learner
        val_1 = vec1.transform(X_val)
        val_2 = vec2.transform(X_val)
        val_3 = vec3.transform(X_val)
        vecs_val = [val_1, val_1, val_2, val_3]

        meta_train, meta_test = [], []
        for (name, bm), Xtr, Xte, Xva in zip(base_models, vecs_tr, vecs_te, vecs_val):
            bm.fit(Xtr, y_train)
            if hasattr(bm, "predict_proba"):
                meta_train.append(bm.predict_proba(Xva))
                meta_test.append(bm.predict_proba(Xte))
            else:
                p_tr = bm.predict(Xva)
                p_te = bm.predict(Xte)
                # One-hot encode
                def onehot(arr):
                    m = np.zeros((len(arr), 3))
                    m[np.arange(len(arr)), arr] = 1
                    return m
                meta_train.append(onehot(p_tr))
                meta_test.append(onehot(p_te))

        M_train = np.hstack(meta_train)
        M_test  = np.hstack(meta_test)

        meta_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
        meta_clf.fit(M_train, y_val)
        y_pred = meta_clf.predict(M_test).tolist()
        log_result(exp_id, "Stacking (NB+CNB+LR → meta-LR)",
                    "char+combined TF-IDF, val-set meta-train",
                    "4 base models → LR meta-learner",
                    y_test, y_pred, time.time() - t0,
                    notes="Meta-learner trained on val set predictions")
    except Exception as e:
        print(f"  [EXP_056] FAILED: {e}")

    # Another stacking variant: train base models on train, use held-out val for meta-train
    try:
        exp_id = "EXP_057"
        t0 = time.time()
        # Use a 5-fold CV approach for meta-features on training set
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

        vec_stk = char_vec((2, 5), 50000)
        # Fit vectorizer on full training set first
        X_tr_v = vec_stk.fit_transform(X_train)
        X_te_v = vec_stk.transform(X_test)

        base_clfs = [
            MultinomialNB(alpha=0.1),
            ComplementNB(alpha=0.1),
            ComplementNB(alpha=0.01),
            LogisticRegression(C=5.0, max_iter=2000, random_state=SEED),
        ]

        meta_tr = np.zeros((len(y_train), 3 * len(base_clfs)))
        for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_tr_v, y_train)):
            for ci, bc in enumerate(base_clfs):
                bc_clone = type(bc)(**bc.get_params())
                bc_clone.fit(X_tr_v[tr_idx], [y_train[i] for i in tr_idx])
                if hasattr(bc_clone, "predict_proba"):
                    p = bc_clone.predict_proba(X_tr_v[va_idx])
                else:
                    p_arr = bc_clone.predict(X_tr_v[va_idx])
                    p = np.zeros((len(p_arr), 3))
                    p[np.arange(len(p_arr)), p_arr] = 1
                meta_tr[va_idx, ci*3:(ci+1)*3] = p

        # Final base models trained on full training set
        meta_te = np.zeros((len(y_test), 3 * len(base_clfs)))
        for ci, bc in enumerate(base_clfs):
            bc.fit(X_tr_v, y_train)
            if hasattr(bc, "predict_proba"):
                p = bc.predict_proba(X_te_v)
            else:
                p_arr = bc.predict(X_te_v)
                p = np.zeros((len(p_arr), 3))
                p[np.arange(len(p_arr)), p_arr] = 1
            meta_te[:, ci*3:(ci+1)*3] = p

        meta_clf = LogisticRegression(C=5.0, max_iter=1000, random_state=SEED)
        meta_clf.fit(meta_tr, y_train)
        y_pred = meta_clf.predict(meta_te).tolist()
        log_result(exp_id, "Stacking 5-fold CV (NB+CNB×2+LR → meta-LR)",
                    "char TF-IDF (2-5), 5-fold CV meta-features",
                    "4 base models, 5-fold CV, C=5 meta-LR",
                    y_test, y_pred, time.time() - t0,
                    notes="Proper CV-based stacking")
    except Exception as e:
        print(f"  [EXP_057] FAILED: {e}")

    # -----------------------------------------------------------------
    print("\n=== L1 regularized LR (sparse, char) ===")
    for exp_id, C_val in [("EXP_058", 1.0), ("EXP_059", 5.0), ("EXP_060", 0.5)]:
        t0 = time.time()
        try:
            vec = char_vec((2, 5), 50000)
            clf = LogisticRegression(C=C_val, penalty="l1", solver="saga",
                                      max_iter=3000, class_weight="balanced",
                                      random_state=SEED)
            X_tr_v = vec.fit_transform(X_train)
            X_te_v = vec.transform(X_test)
            clf.fit(X_tr_v, y_train)
            y_pred = clf.predict(X_te_v)
            log_result(exp_id, f"LR-L1 (saga, C={C_val}, balanced, char)",
                        "char TF-IDF (2-5), max=50k",
                        f"C={C_val}, L1, saga, balanced",
                        y_test, y_pred, time.time() - t0)
        except Exception as e:
            print(f"  [{exp_id}] FAILED: {e}")

    # -----------------------------------------------------------------
    # Save and visualize
    df = save_log("2")

    if len(df) > 0:
        print(f"\n{'='*65}")
        print(f"  Phase 2 Best Results (top-10)")
        print(f"{'='*65}")
        for _, row in df.head(10).iterrows():
            d = row["F1_macro"] - NB_BASELINE
            s = "↑" if d >= 0 else "↓"
            print(f"  {row['Experiment_ID']:>8}  {row['Model'][:35]:<35}  "
                  f"F1={row['F1_macro']:.4f}  {s}{abs(d):.4f}")
        print(f"{'='*65}")
        print(f"\n  Best: {df.iloc[0]['Model']} @ F1={df.iloc[0]['F1_macro']:.4f}")
        print(f"  Improvement over baseline: +{df.iloc[0]['F1_macro']-NB_BASELINE:.4f}")


if __name__ == "__main__":
    main()
