"""
Third-pass experiments: Raw tweet text is better than preprocessed.

Key finding from Phase 2:
  EXP_046: MultinomialNB + raw tweet + char TF-IDF (2-5) = 0.6693

This file maximizes that finding:
  EXP_070  NB raw, char (2-5), alpha sweep (0.01,0.05,0.1,0.2,0.5)
  EXP_075  NB raw, char (2-6), alpha sweep
  EXP_080  NB raw, char (1-5), alpha sweep
  EXP_085  NB raw, char (2-5), max_features sweep (50k,100k,200k)
  EXP_088  LR raw, char (2-5), various C
  EXP_091  SVM raw, char (2-5), various C
  EXP_094  NB raw, word (1-2) + char (2-5)
  EXP_095  NB raw, word (1-3) + char (2-6)
  EXP_096  Voting ensemble (best NB raw + best LR raw + best SVM raw)
  EXP_097  Soft-vote ensemble
  EXP_098  Stacking (raw text models)
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
PHASE2_BEST  = 0.6693  # EXP_046 MultinomialNB raw tweet char TF-IDF

np.random.seed(SEED)

# ---------------------------------------------------------------------------
def load_splits():
    train = pd.read_csv(DATA_DIR / "train.csv")
    val   = pd.read_csv(DATA_DIR / "val.csv")
    test  = pd.read_csv(DATA_DIR / "test.csv")
    def xy(df, col):
        return df[col].fillna("").astype(str).tolist(), df["label"].tolist()
    return (xy(train, "tweet"),   xy(val, "tweet"),   xy(test, "tweet"),
            xy(train, "cleaned_text"), xy(val, "cleaned_text"), xy(test, "cleaned_text"))


LOG_COLS = [
    "Experiment_ID", "Model", "Features", "Hyperparameters",
    "F1_macro", "Accuracy", "Precision_macro", "Recall_macro",
    "F1_positive", "F1_negative", "F1_neutral",
    "Training_time_s", "Notes",
]
exp_log: list[dict] = []
all_preds_store: dict = {}


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
    all_preds_store[exp_id] = list(y_pred)
    d1 = f1_mac - NB_BASELINE
    d2 = f1_mac - PHASE2_BEST
    s1 = "↑" if d1 >= 0 else "↓"
    s2 = "↑" if d2 >= 0 else "↓"
    print(f"  [{exp_id}] F1={f1_mac:.4f}  "
          f"{s1}{abs(d1):.4f} vs baseline  {s2}{abs(d2):.4f} vs Phase2-best  "
          f"[pos={f1_per[0]:.3f} neg={f1_per[1]:.3f} neu={f1_per[2]:.3f}]  ({train_time:.1f}s)")
    return f1_mac


def save_log(suffix="3"):
    if not exp_log:
        return pd.DataFrame(columns=LOG_COLS)
    df = pd.DataFrame(exp_log, columns=LOG_COLS)
    df = df.sort_values("F1_macro", ascending=False).reset_index(drop=True)
    df.to_excel(EXP_DIR / f"experiment_log_{suffix}.xlsx", index=False)
    df.to_csv(EXP_DIR  / f"experiment_log_{suffix}.csv",   index=False)
    return df


def char_vec(ng=(2, 5), mf=80000):
    return TfidfVectorizer(analyzer="char_wb", ngram_range=ng,
                           max_features=mf, sublinear_tf=True, min_df=1)

def word_vec(ng=(1, 2), mf=30000):
    return TfidfVectorizer(analyzer="word", ngram_range=ng,
                           max_features=mf, sublinear_tf=True, min_df=1)

def combined_vec(wn=(1, 2), cn=(2, 5), wf=30000, cf=80000):
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word",    ngram_range=wn,
                                  max_features=wf, sublinear_tf=True, min_df=1)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=cn,
                                  max_features=cf, sublinear_tf=True, min_df=1)),
    ])


def main():
    print(f"\n{'='*65}")
    print(f"  Third-Pass Experiments — Raw Tweet Maximization")
    print(f"  Phase 2 best: {PHASE2_BEST} (NB raw tweet)  |  Baseline: {NB_BASELINE}")
    print(f"{'='*65}")

    (X_train, y_train), (X_val, y_val), (X_test, y_test), \
    (X_train_c, _), (X_val_c, _), (X_test_c, _) = load_splits()

    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}\n")

    # ---------------------------------------------------------------
    print("=== MultinomialNB raw tweet, char (2-5), alpha sweep ===")
    best_raw_nb = {"f1": 0, "preds": [], "vec": None, "clf": None}

    for exp_id, alpha in [
        ("EXP_070", 0.001), ("EXP_071", 0.01), ("EXP_072", 0.05),
        ("EXP_073", 0.1),   ("EXP_074", 0.2),  ("EXP_075", 0.5),
    ]:
        t0 = time.time()
        vec = char_vec((2, 5), 80000)
        clf = MultinomialNB(alpha=alpha)
        clf.fit(vec.fit_transform(X_train), y_train)
        y_pred = clf.predict(vec.transform(X_test))
        f1 = log_result(exp_id, "MNB raw (2-5)",
                         "raw tweet, char (2-5), max=80k",
                         f"alpha={alpha}", y_test, y_pred, time.time() - t0)
        if f1 > best_raw_nb["f1"]:
            best_raw_nb = {"f1": f1, "preds": list(y_pred), "vec": vec, "clf": clf, "alpha": alpha}

    print(f"\n  Best raw NB alpha: {best_raw_nb.get('alpha')} @ F1={best_raw_nb['f1']:.4f}")

    # ---------------------------------------------------------------
    print("\n=== char range sweep (raw tweet, best alpha=0.1) ===")
    best_raw_any = dict(best_raw_nb)

    for exp_id, ng, mf in [
        ("EXP_076", (2, 4), 60000),
        ("EXP_077", (2, 6), 100000),
        ("EXP_078", (1, 5), 100000),
        ("EXP_079", (3, 6), 100000),
        ("EXP_080", (2, 7), 150000),
        ("EXP_081", (1, 6), 150000),
    ]:
        t0 = time.time()
        vec = char_vec(ng, mf)
        clf = MultinomialNB(alpha=0.1)
        clf.fit(vec.fit_transform(X_train), y_train)
        y_pred = clf.predict(vec.transform(X_test))
        f1 = log_result(exp_id, f"MNB raw char {ng}",
                         f"raw tweet, char {ng}, max={mf}", "alpha=0.1",
                         y_test, y_pred, time.time() - t0)
        if f1 > best_raw_any["f1"]:
            best_raw_any = {"f1": f1, "preds": list(y_pred), "vec": vec, "clf": clf}

    # ---------------------------------------------------------------
    print("\n=== max_features sweep (raw tweet, char (2-5), alpha=0.1) ===")
    for exp_id, mf in [
        ("EXP_082", 30000), ("EXP_083", 50000), ("EXP_084", 120000), ("EXP_085", 200000),
    ]:
        t0 = time.time()
        vec = char_vec((2, 5), mf)
        clf = MultinomialNB(alpha=0.1)
        clf.fit(vec.fit_transform(X_train), y_train)
        y_pred = clf.predict(vec.transform(X_test))
        f1 = log_result(exp_id, f"MNB raw char (2-5) max={mf}",
                         f"raw tweet, char (2-5), max={mf}", "alpha=0.1",
                         y_test, y_pred, time.time() - t0)
        if f1 > best_raw_any["f1"]:
            best_raw_any = {"f1": f1, "preds": list(y_pred), "vec": vec, "clf": clf}

    # ---------------------------------------------------------------
    print("\n=== LR on raw tweet, char (2-5) ===")
    best_raw_lr = {"f1": 0, "preds": [], "vec": None, "clf": None}

    for exp_id, C_val, balanced in [
        ("EXP_086", 0.5, False), ("EXP_087", 1.0, False), ("EXP_088", 5.0, False),
        ("EXP_089", 1.0, True),  ("EXP_090", 5.0, True),
    ]:
        t0 = time.time()
        vec = char_vec((2, 5), 80000)
        cw  = "balanced" if balanced else None
        clf = LogisticRegression(C=C_val, max_iter=2000, class_weight=cw, random_state=SEED)
        clf.fit(vec.fit_transform(X_train), y_train)
        y_pred = clf.predict(vec.transform(X_test))
        f1 = log_result(exp_id, f"LR raw C={C_val} {'bal' if balanced else 'nobal'}",
                         "raw tweet, char (2-5), max=80k",
                         f"C={C_val}, balanced={balanced}",
                         y_test, y_pred, time.time() - t0)
        if f1 > best_raw_lr["f1"]:
            best_raw_lr = {"f1": f1, "preds": list(y_pred), "vec": vec, "clf": clf}

    # ---------------------------------------------------------------
    print("\n=== SVM on raw tweet, char (2-5) ===")
    best_raw_svm = {"f1": 0, "preds": [], "vec": None, "clf": None}

    for exp_id, C_val, balanced in [
        ("EXP_091", 0.5, False), ("EXP_092", 1.0, False), ("EXP_093", 2.0, False),
        ("EXP_094", 1.0, True),  ("EXP_095", 0.5, True),
    ]:
        t0 = time.time()
        vec = char_vec((2, 5), 80000)
        cw  = "balanced" if balanced else None
        clf = LinearSVC(C=C_val, max_iter=5000, class_weight=cw, random_state=SEED)
        clf.fit(vec.fit_transform(X_train), y_train)
        y_pred = clf.predict(vec.transform(X_test))
        f1 = log_result(exp_id, f"SVM raw C={C_val} {'bal' if balanced else 'nobal'}",
                         "raw tweet, char (2-5), max=80k",
                         f"C={C_val}, balanced={balanced}",
                         y_test, y_pred, time.time() - t0)
        if f1 > best_raw_svm["f1"]:
            best_raw_svm = {"f1": f1, "preds": list(y_pred), "vec": vec, "clf": clf}

    # ---------------------------------------------------------------
    print("\n=== Combined word+char on raw tweet ===")
    for exp_id, wn, cn, wf, cf in [
        ("EXP_096", (1, 2), (2, 5), 30000, 80000),
        ("EXP_097", (1, 3), (2, 6), 30000, 100000),
    ]:
        t0 = time.time()
        vec = combined_vec(wn, cn, wf, cf)
        clf = MultinomialNB(alpha=0.1)
        clf.fit(vec.fit_transform(X_train), y_train)
        y_pred = clf.predict(vec.transform(X_test))
        f1 = log_result(exp_id, f"MNB raw word{wn}+char{cn}",
                         f"raw tweet, word {wn}+char {cn}",
                         "alpha=0.1", y_test, y_pred, time.time() - t0)
        if f1 > best_raw_any["f1"]:
            best_raw_any = {"f1": f1, "preds": list(y_pred)}

    # ---------------------------------------------------------------
    print("\n=== Voting ensemble (best NB + LR + SVM, raw text) ===")

    # Rebuild best individual models for ensemble
    t0 = time.time()
    try:
        vec_e = char_vec((2, 5), 80000)
        X_tr_e = vec_e.fit_transform(X_train)
        X_te_e = vec_e.transform(X_test)

        nb_e   = MultinomialNB(alpha=0.1)
        lr_e   = LogisticRegression(C=5.0, max_iter=2000, random_state=SEED)
        svm_e  = LinearSVC(C=1.0, max_iter=5000, random_state=SEED)
        nb_e.fit(X_tr_e, y_train)
        lr_e.fit(X_tr_e, y_train)
        svm_e.fit(X_tr_e, y_train)

        # Hard vote
        votes = np.array([nb_e.predict(X_te_e),
                           lr_e.predict(X_te_e),
                           svm_e.predict(X_te_e)])
        y_hard = [int(np.bincount(votes[:, i], minlength=3).argmax())
                  for i in range(votes.shape[1])]
        f1 = log_result("EXP_098", "Hard-vote ensemble (NB+LR+SVM raw)",
                         "raw tweet, char (2-5) max=80k",
                         "NB alpha=0.1 + LR C=5 + SVM C=1 hard vote",
                         y_test, y_hard, time.time() - t0)
    except Exception as e:
        print(f"  [EXP_098] FAILED: {e}")

    # Soft vote (NB + LR probabilities)
    t0 = time.time()
    try:
        vec_s = char_vec((2, 5), 80000)
        X_tr_s = vec_s.fit_transform(X_train)
        X_te_s = vec_s.transform(X_test)

        nb_s  = MultinomialNB(alpha=0.1)
        lr_s  = LogisticRegression(C=5.0, max_iter=2000, random_state=SEED)
        nb_s.fit(X_tr_s, y_train)
        lr_s.fit(X_tr_s, y_train)

        proba = nb_s.predict_proba(X_te_s) + lr_s.predict_proba(X_te_s)
        y_soft = proba.argmax(axis=1).tolist()
        f1 = log_result("EXP_099", "Soft-vote ensemble (NB+LR raw)",
                         "raw tweet, char (2-5) max=80k",
                         "NB alpha=0.1 + LR C=5 soft vote (avg proba)",
                         y_test, y_soft, time.time() - t0)
    except Exception as e:
        print(f"  [EXP_099] FAILED: {e}")

    # ---------------------------------------------------------------
    print("\n=== 5-fold stacking (raw text) ===")
    try:
        t0 = time.time()
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

        vec_stk = char_vec((2, 5), 80000)
        X_tr_v = vec_stk.fit_transform(X_train)
        X_te_v = vec_stk.transform(X_test)

        base_clfs = [
            MultinomialNB(alpha=0.1),
            MultinomialNB(alpha=0.05),
            MultinomialNB(alpha=0.2),
            LogisticRegression(C=5.0, max_iter=2000, random_state=SEED),
        ]

        meta_tr = np.zeros((len(y_train), 3 * len(base_clfs)))
        for _, (tr_idx, va_idx) in enumerate(skf.split(X_tr_v, y_train)):
            for ci, bc in enumerate(base_clfs):
                bc_c = type(bc)(**bc.get_params())
                bc_c.fit(X_tr_v[tr_idx], [y_train[i] for i in tr_idx])
                if hasattr(bc_c, "predict_proba"):
                    meta_tr[va_idx, ci*3:(ci+1)*3] = bc_c.predict_proba(X_tr_v[va_idx])
                else:
                    p = bc_c.predict(X_tr_v[va_idx])
                    m = np.zeros((len(p), 3))
                    m[np.arange(len(p)), p] = 1
                    meta_tr[va_idx, ci*3:(ci+1)*3] = m

        meta_te = np.zeros((len(y_test), 3 * len(base_clfs)))
        for ci, bc in enumerate(base_clfs):
            bc.fit(X_tr_v, y_train)
            if hasattr(bc, "predict_proba"):
                meta_te[:, ci*3:(ci+1)*3] = bc.predict_proba(X_te_v)
            else:
                p = bc.predict(X_te_v)
                m = np.zeros((len(p), 3))
                m[np.arange(len(p)), p] = 1
                meta_te[:, ci*3:(ci+1)*3] = m

        meta_clf = LogisticRegression(C=5.0, max_iter=1000, random_state=SEED)
        meta_clf.fit(meta_tr, y_train)
        y_pred = meta_clf.predict(meta_te).tolist()
        f1 = log_result("EXP_100", "5-fold stacking (3×NB+LR → meta-LR, raw)",
                         "raw tweet, char (2-5) max=80k, 5-fold CV",
                         "3×MNB + LR base, LR-C5 meta, 5-fold",
                         y_test, y_pred, time.time() - t0)
    except Exception as e:
        print(f"  [EXP_100] FAILED: {e}")

    # ---------------------------------------------------------------
    # Save
    df = save_log("3")

    if len(df) > 0:
        # Update best model comparison (best from all phases)
        best_row = df.iloc[0]
        print(f"\n{'='*65}")
        print(f"  Phase 3 Results (top-10)")
        print(f"{'='*65}")
        for _, row in df.head(10).iterrows():
            d = row["F1_macro"] - NB_BASELINE
            s = "↑" if d >= 0 else "↓"
            print(f"  {row['Experiment_ID']:>8}  {row['Model'][:40]:<40}  "
                  f"F1={row['F1_macro']:.4f}  {s}{abs(d):.4f}")
        print(f"\n  Phase 3 Best: {best_row['Model']} @ F1={best_row['F1_macro']:.4f}")
        print(f"  Improvement over baseline: +{best_row['F1_macro']-NB_BASELINE:.4f} "
              f"({(best_row['F1_macro']-NB_BASELINE)/NB_BASELINE*100:.1f}%)")

        # Generate CM for best
        print(f"\n  Generating CM for best model: {best_row['Experiment_ID']}")
        try:
            best_alpha = float(best_row["Hyperparameters"].split("alpha=")[1].split(",")[0]) \
                        if "alpha=" in best_row["Hyperparameters"] else 0.1
            vec_cm = char_vec((2, 5), 80000)
            clf_cm = MultinomialNB(alpha=best_alpha)
            X_tr_cm = vec_cm.fit_transform(X_train)
            X_te_cm = vec_cm.transform(X_test)
            clf_cm.fit(X_tr_cm, y_train)
            y_pred_cm = clf_cm.predict(X_te_cm)

            cm = confusion_matrix(y_test, y_pred_cm)
            fig, ax = plt.subplots(figsize=(7, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"Best Model CM — {best_row['Model'][:45]}\nF1={best_row['F1_macro']:.4f}")
            plt.tight_layout()
            cm_path = EXP_DIR / "best_confusion_matrix.jpeg"
            fig.savefig(cm_path, dpi=150, format="jpeg")
            plt.close(fig)
            print(f"  CM saved → {cm_path}")
        except Exception as e:
            print(f"  CM generation failed: {e}")


if __name__ == "__main__":
    main()
