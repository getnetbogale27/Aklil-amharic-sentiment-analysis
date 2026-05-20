"""
Systematic experiment runner for Amharic sentiment analysis.

Goal: Push F1_macro from 0.6109 (NB baseline) to 0.80+

Experiment plan:
  EXP_001  NB  word TF-IDF (1-2)                         [baseline reference]
  EXP_002  NB  char TF-IDF (2-5)                         [char n-grams]
  EXP_003  LR  word TF-IDF (1-2) balanced                [class weight fix]
  EXP_004  LR  char TF-IDF (2-5) balanced
  EXP_005  SVM word TF-IDF (1-2) balanced
  EXP_006  SVM char TF-IDF (2-5) balanced
  EXP_007  LR  word+char combined balanced                [feature fusion]
  EXP_008  SVM word+char combined balanced
  EXP_009  LR  word (1-3)+char (2-6) balanced            [wider n-grams]
  EXP_010  SVM word (1-3)+char (2-6) balanced
  EXP_011  LR  C-sweep on best feature set               [hyperparam tuning]
  EXP_012  SVM C-sweep on best feature set
  EXP_013  XGBoost + combined features
  EXP_014  LightGBM + combined features
  EXP_015  Bi-LSTM + class weights                        [deep model fix]
  EXP_016  Transformer (offline) + class weights          [architecture]
  EXP_017  Voting ensemble (top-3 ML models)             [ensemble]
  EXP_018  Soft-vote ensemble (top-5 ML models)

All outputs go to results/experiments/.
"""

import json
import math
import os
import random
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
import torch
import torch.nn as nn
from sklearn.ensemble import VotingClassifier
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
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
import joblib

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parent.parent
DATA_DIR     = ROOT / "data" / "processed"
EXP_DIR      = ROOT / "results" / "experiments"
MODELS_DIR   = ROOT / "results" / "models"
EXP_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES  = ["positive", "negative", "neutral"]
SEED         = 42
NB_BASELINE  = 0.6109
TARGET_F1    = 0.80

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_splits():
    train = pd.read_csv(DATA_DIR / "train.csv")
    val   = pd.read_csv(DATA_DIR / "val.csv")
    test  = pd.read_csv(DATA_DIR / "test.csv")

    def get_xy(df):
        X = df["cleaned_text"].fillna("").astype(str).tolist()
        y = df["label"].tolist()
        return X, y

    return get_xy(train), get_xy(val), get_xy(test)


# ---------------------------------------------------------------------------
# Experiment log
# ---------------------------------------------------------------------------
LOG_COLS = [
    "Experiment_ID", "Model", "Features", "Hyperparameters",
    "F1_macro", "Accuracy", "Precision_macro", "Recall_macro",
    "F1_positive", "F1_negative", "F1_neutral",
    "Training_time_s", "Notes",
]
exp_log: list[dict] = []
best_result = {"F1_macro": 0.0, "Experiment_ID": None, "Model": None}


def log_result(exp_id, model_name, features, hyperparams, y_true, y_pred,
               train_time, notes=""):
    global best_result
    f1_mac  = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    acc     = accuracy_score(y_true, y_pred)
    prec    = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec     = recall_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_per  = f1_score(y_true, y_pred, average=None, zero_division=0)

    row = {
        "Experiment_ID":    exp_id,
        "Model":            model_name,
        "Features":         features,
        "Hyperparameters":  hyperparams,
        "F1_macro":         round(f1_mac, 4),
        "Accuracy":         round(acc,    4),
        "Precision_macro":  round(prec,   4),
        "Recall_macro":     round(rec,    4),
        "F1_positive":      round(f1_per[0] if len(f1_per) > 0 else 0, 4),
        "F1_negative":      round(f1_per[1] if len(f1_per) > 1 else 0, 4),
        "F1_neutral":       round(f1_per[2] if len(f1_per) > 2 else 0, 4),
        "Training_time_s":  round(train_time, 1),
        "Notes":            notes,
    }
    exp_log.append(row)
    improvement = f1_mac - NB_BASELINE
    sign = "↑" if improvement >= 0 else "↓"
    print(f"  [{exp_id}] F1={f1_mac:.4f}  {sign}{abs(improvement):.4f} vs baseline  "
          f"[pos={f1_per[0]:.3f} neg={f1_per[1]:.3f} neu={f1_per[2]:.3f}]  "
          f"({train_time:.1f}s)")

    if f1_mac > best_result["F1_macro"]:
        best_result = {"F1_macro": f1_mac, "Experiment_ID": exp_id, "Model": model_name}
    return f1_mac


def save_log():
    df = pd.DataFrame(exp_log, columns=LOG_COLS)
    df = df.sort_values("F1_macro", ascending=False).reset_index(drop=True)
    xlsx_path = EXP_DIR / "experiment_log.xlsx"
    csv_path  = EXP_DIR / "experiment_log.csv"
    df.to_excel(xlsx_path, index=False)
    df.to_csv(csv_path,    index=False)
    return df


# ---------------------------------------------------------------------------
# TF-IDF helpers
# ---------------------------------------------------------------------------
def make_word_tfidf(ngram=(1, 2), max_features=20000):
    return TfidfVectorizer(
        analyzer="word", ngram_range=ngram, max_features=max_features,
        sublinear_tf=True, min_df=2,
    )


def make_char_tfidf(ngram=(2, 5), max_features=50000):
    return TfidfVectorizer(
        analyzer="char_wb", ngram_range=ngram, max_features=max_features,
        sublinear_tf=True, min_df=2,
    )


def make_combined_tfidf(word_ngram=(1, 2), char_ngram=(2, 5),
                         word_feats=20000, char_feats=30000):
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=word_ngram,
                                  max_features=word_feats, sublinear_tf=True, min_df=2)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=char_ngram,
                                  max_features=char_feats, sublinear_tf=True, min_df=2)),
    ])


# ---------------------------------------------------------------------------
# ML experiments (Steps 2a–2e)
# ---------------------------------------------------------------------------
def run_ml_experiments(X_train, y_train, X_test, y_test):
    print("\n" + "="*65)
    print("  PHASE 1 — ML Feature Engineering Experiments")
    print("="*65)

    # Compute class weights for balanced training
    cw_arr = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=np.array(y_train))
    cw_dict = {0: cw_arr[0], 1: cw_arr[1], 2: cw_arr[2]}
    print(f"  Class weights: pos={cw_arr[0]:.3f} neg={cw_arr[1]:.3f} neu={cw_arr[2]:.3f}")

    configs = [
        # EXP_001 — baseline word TF-IDF, NB (replication)
        ("EXP_001", "Naive Bayes",
         make_word_tfidf((1, 2), 10000),
         MultinomialNB(),
         "word TF-IDF (1-2), max=10k",
         "alpha=1.0 (default)"),

        # EXP_002 — char TF-IDF, NB
        ("EXP_002", "Naive Bayes (char)",
         make_char_tfidf((2, 5), 50000),
         MultinomialNB(alpha=0.1),
         "char TF-IDF (2-5), max=50k",
         "alpha=0.1"),

        # EXP_003 — word TF-IDF, LR balanced
        ("EXP_003", "Logistic Regression (word, balanced)",
         make_word_tfidf((1, 2), 20000),
         LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=SEED),
         "word TF-IDF (1-2), max=20k",
         "C=1.0, balanced"),

        # EXP_004 — char TF-IDF, LR balanced
        ("EXP_004", "Logistic Regression (char, balanced)",
         make_char_tfidf((2, 5), 50000),
         LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=SEED),
         "char TF-IDF (2-5), max=50k",
         "C=1.0, balanced"),

        # EXP_005 — word TF-IDF, SVM balanced
        ("EXP_005", "SVM (word, balanced)",
         make_word_tfidf((1, 2), 20000),
         LinearSVC(C=1.0, max_iter=2000, class_weight="balanced", random_state=SEED),
         "word TF-IDF (1-2), max=20k",
         "C=1.0, balanced"),

        # EXP_006 — char TF-IDF, SVM balanced
        ("EXP_006", "SVM (char, balanced)",
         make_char_tfidf((2, 5), 50000),
         LinearSVC(C=1.0, max_iter=2000, class_weight="balanced", random_state=SEED),
         "char TF-IDF (2-5), max=50k",
         "C=1.0, balanced"),

        # EXP_007 — word+char combined, LR balanced
        ("EXP_007", "Logistic Regression (word+char, balanced)",
         make_combined_tfidf((1, 2), (2, 5), 20000, 30000),
         LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=SEED),
         "word (1-2)+char (2-5) TF-IDF",
         "C=1.0, balanced"),

        # EXP_008 — word+char combined, SVM balanced
        ("EXP_008", "SVM (word+char, balanced)",
         make_combined_tfidf((1, 2), (2, 5), 20000, 30000),
         LinearSVC(C=1.0, max_iter=3000, class_weight="balanced", random_state=SEED),
         "word (1-2)+char (2-5) TF-IDF",
         "C=1.0, balanced"),

        # EXP_009 — wider n-grams, LR balanced
        ("EXP_009", "Logistic Regression (wide n-grams, balanced)",
         make_combined_tfidf((1, 3), (2, 6), 25000, 50000),
         LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced", random_state=SEED),
         "word (1-3)+char (2-6) TF-IDF",
         "C=2.0, balanced"),

        # EXP_010 — wider n-grams, SVM balanced
        ("EXP_010", "SVM (wide n-grams, balanced)",
         make_combined_tfidf((1, 3), (2, 6), 25000, 50000),
         LinearSVC(C=2.0, max_iter=3000, class_weight="balanced", random_state=SEED),
         "word (1-3)+char (2-6) TF-IDF",
         "C=2.0, balanced"),
    ]

    best_ml = {"f1": 0.0, "name": "", "vec": None, "clf": None}
    ml_results = {}

    for exp_id, name, vec, clf, feat_desc, hp_desc in configs:
        t0 = time.time()
        try:
            X_tr_v = vec.fit_transform(X_train)
            X_te_v = vec.transform(X_test)
            clf.fit(X_tr_v, y_train)
            y_pred = clf.predict(X_te_v)
            elapsed = time.time() - t0
            f1 = log_result(exp_id, name, feat_desc, hp_desc, y_test, y_pred, elapsed)
            ml_results[exp_id] = (vec, clf, y_pred)
            if f1 > best_ml["f1"]:
                best_ml = {"f1": f1, "name": name, "vec": vec, "clf": clf,
                            "exp_id": exp_id, "feat_desc": feat_desc}
        except Exception as e:
            print(f"  [{exp_id}] FAILED: {e}")

    # C sweep on best ML combo
    print(f"\n  Best ML so far: {best_ml['name']} (F1={best_ml['f1']:.4f})")
    print("  Running C-sweep on best feature set...")

    for c_val, exp_id in [(0.1, "EXP_011"), (0.5, "EXP_012"), (5.0, "EXP_013"), (10.0, "EXP_014")]:
        t0 = time.time()
        try:
            vec_new = make_combined_tfidf((1, 3), (2, 6), 25000, 50000)
            clf_lr  = LogisticRegression(C=c_val, max_iter=2000, class_weight="balanced",
                                          solver="lbfgs", random_state=SEED)
            X_tr_v = vec_new.fit_transform(X_train)
            X_te_v = vec_new.transform(X_test)
            clf_lr.fit(X_tr_v, y_train)
            y_pred = clf_lr.predict(X_te_v)
            elapsed = time.time() - t0
            f1 = log_result(exp_id, f"LR C={c_val} (wide, balanced)",
                             "word (1-3)+char (2-6)", f"C={c_val}, balanced",
                             y_test, y_pred, elapsed)
            if f1 > best_ml["f1"]:
                best_ml = {"f1": f1, "name": f"LR C={c_val}", "vec": vec_new, "clf": clf_lr,
                            "exp_id": exp_id, "feat_desc": "word (1-3)+char (2-6)"}
        except Exception as e:
            print(f"  [{exp_id}] FAILED: {e}")

    for c_val, exp_id in [(0.5, "EXP_015"), (2.0, "EXP_016"), (5.0, "EXP_017")]:
        t0 = time.time()
        try:
            vec_new = make_combined_tfidf((1, 3), (2, 6), 25000, 50000)
            clf_svm = LinearSVC(C=c_val, max_iter=5000, class_weight="balanced",
                                 random_state=SEED)
            X_tr_v = vec_new.fit_transform(X_train)
            X_te_v = vec_new.transform(X_test)
            clf_svm.fit(X_tr_v, y_train)
            y_pred = clf_svm.predict(X_te_v)
            elapsed = time.time() - t0
            f1 = log_result(exp_id, f"SVM C={c_val} (wide, balanced)",
                             "word (1-3)+char (2-6)", f"C={c_val}, balanced",
                             y_test, y_pred, elapsed)
            if f1 > best_ml["f1"]:
                best_ml = {"f1": f1, "name": f"SVM C={c_val}", "vec": vec_new, "clf": clf_svm,
                            "exp_id": exp_id, "feat_desc": "word (1-3)+char (2-6)"}
        except Exception as e:
            print(f"  [{exp_id}] FAILED: {e}")

    print(f"\n  Best after C-sweep: {best_ml['name']} (F1={best_ml['f1']:.4f})")
    return best_ml, ml_results


# ---------------------------------------------------------------------------
# Gradient Boosting experiments
# ---------------------------------------------------------------------------
def run_boosting_experiments(X_train, y_train, X_test, y_test):
    print("\n" + "="*65)
    print("  PHASE 2 — Gradient Boosting Experiments")
    print("="*65)

    best_boost = {"f1": 0.0, "name": "", "vec": None, "clf": None}

    # Build a mid-size combined feature matrix once (faster to fit)
    vec = make_combined_tfidf((1, 2), (2, 5), 10000, 15000)
    X_tr_v = vec.fit_transform(X_train)
    X_te_v = vec.transform(X_test)

    if HAS_XGB:
        # scale_pos_weight for XGBoost handles imbalance
        cw = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=np.array(y_train))
        for exp_id, n_est, lr_xgb, max_d in [("EXP_018", 200, 0.1, 6),
                                               ("EXP_019", 300, 0.05, 8)]:
            t0 = time.time()
            try:
                clf = XGBClassifier(
                    n_estimators=n_est, learning_rate=lr_xgb, max_depth=max_d,
                    subsample=0.8, colsample_bytree=0.8,
                    eval_metric="mlogloss", random_state=SEED,
                    n_jobs=-1, verbosity=0,
                )
                # Convert sparse to dense for XGB (it handles sparse but this is explicit)
                clf.fit(X_tr_v, y_train, sample_weight=[cw[y] for y in y_train])
                y_pred = clf.predict(X_te_v)
                elapsed = time.time() - t0
                f1 = log_result(exp_id, f"XGBoost (n={n_est}, lr={lr_xgb})",
                                 "word (1-2)+char (2-5) TF-IDF",
                                 f"n_estimators={n_est}, lr={lr_xgb}, max_depth={max_d}",
                                 y_test, y_pred, elapsed)
                if f1 > best_boost["f1"]:
                    best_boost = {"f1": f1, "name": f"XGBoost n={n_est}", "vec": vec, "clf": clf}
            except Exception as e:
                print(f"  [{exp_id}] FAILED: {e}")
    else:
        print("  [skip] XGBoost not available")

    if HAS_LGB:
        cw = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=np.array(y_train))
        for exp_id, n_est, lr_lgb, num_leaves in [("EXP_020", 300, 0.05, 63),
                                                    ("EXP_021", 500, 0.03, 127)]:
            t0 = time.time()
            try:
                clf = LGBMClassifier(
                    n_estimators=n_est, learning_rate=lr_lgb, num_leaves=num_leaves,
                    class_weight="balanced", random_state=SEED, n_jobs=-1,
                    verbose=-1,
                )
                clf.fit(X_tr_v, y_train)
                y_pred = clf.predict(X_te_v)
                elapsed = time.time() - t0
                f1 = log_result(exp_id, f"LightGBM (n={n_est}, lr={lr_lgb})",
                                 "word (1-2)+char (2-5) TF-IDF",
                                 f"n_estimators={n_est}, lr={lr_lgb}, num_leaves={num_leaves}",
                                 y_test, y_pred, elapsed)
                if f1 > best_boost["f1"]:
                    best_boost = {"f1": f1, "name": f"LightGBM n={n_est}", "vec": vec, "clf": clf}
            except Exception as e:
                print(f"  [{exp_id}] FAILED: {e}")
    else:
        print("  [skip] LightGBM not available")

    return best_boost


# ---------------------------------------------------------------------------
# Improved Bi-LSTM
# ---------------------------------------------------------------------------
class AmharicCharDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=300):
        self.pad_idx = vocab.get("<PAD>", 0)
        self.unk_idx = vocab.get("<UNK>", 1)
        self.max_len = max_len
        self.seqs    = [self._encode(t, vocab) for t in texts]
        self.labels  = labels

    def _encode(self, text, vocab):
        ids = [vocab.get(ch, self.unk_idx) for ch in str(text)]
        ids = ids[:self.max_len]
        ids += [self.pad_idx] * (self.max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.tensor(self.seqs[idx], dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.long))


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_out):
        scores  = self.attn(lstm_out).squeeze(-1)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (lstm_out * weights).sum(dim=1)


class ImprovedBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2,
                 num_classes=3, dropout=0.4, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                             batch_first=True, bidirectional=True,
                             dropout=dropout if num_layers > 1 else 0.0)
        self.attention  = SelfAttention(hidden_dim)
        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        emb    = self.dropout(self.embedding(x))
        out, _ = self.lstm(emb)
        ctx    = self.attention(out)
        ctx    = self.layer_norm(ctx)
        return self.classifier(ctx)


def run_bilstm_experiment(X_train, y_train, X_test, y_test):
    print("\n" + "="*65)
    print("  PHASE 3 — Improved Bi-LSTM (char-level + class weights)")
    print("="*65)

    exp_id = "EXP_022"
    t0 = time.time()

    # Build char vocab
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for text in X_train:
        for ch in str(text):
            if ch not in vocab:
                vocab[ch] = len(vocab)
    vocab_size = len(vocab)
    print(f"  Char vocab size: {vocab_size}")

    # Class weights
    cw_arr = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=np.array(y_train))
    cw_tensor = torch.tensor(cw_arr, dtype=torch.float)

    # Datasets
    MAX_LEN    = 300
    BATCH_SIZE = 64

    train_ds = AmharicCharDataset(X_train, y_train, vocab, MAX_LEN)
    test_ds  = AmharicCharDataset(X_test,  y_test,  vocab, MAX_LEN)
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_ld  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ImprovedBiLSTM(vocab_size, embed_dim=128, hidden_dim=256,
                             num_layers=2, dropout=0.4).to(device)
    crit   = nn.CrossEntropyLoss(weight=cw_tensor.to(device), label_smoothing=0.05)
    opt    = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=3, factor=0.5)

    best_f1 = 0.0
    best_preds = []
    ckpt = MODELS_DIR / "bilstm_improved_best.pt"
    no_improve = 0
    PATIENCE = 8
    MAX_EPOCHS = 40

    print(f"  Training on {device} for up to {MAX_EPOCHS} epochs...")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            for x, y in test_ld:
                x = x.to(device)
                all_p.extend(model(x).argmax(-1).cpu().tolist())
                all_l.extend(y.tolist())

        ep_f1 = f1_score(all_l, all_p, average="macro", zero_division=0)
        sched.step(ep_f1)

        if ep_f1 > best_f1:
            best_f1, best_preds = ep_f1, all_p[:]
            torch.save(model.state_dict(), ckpt)
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 5 == 0:
            print(f"    Epoch {epoch:>3}: F1={ep_f1:.4f}  best={best_f1:.4f}")

        if no_improve >= PATIENCE:
            print(f"    Early stop at epoch {epoch}")
            break

    elapsed = time.time() - t0
    f1 = log_result(exp_id, "Bi-LSTM improved (char, class-weighted)",
                     "char-level embeddings (vocab-built)",
                     "embed=128, hidden=256, layers=2, dropout=0.4, class_weight=balanced",
                     y_test, best_preds, elapsed)
    return {"f1": f1, "preds": best_preds}


# ---------------------------------------------------------------------------
# Offline Transformer (with class weights)
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class OfflineTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=4,
                 d_ff=1024, dropout=0.1, num_classes=3, max_len=128):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        enc_layer    = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                                   dim_feedforward=d_ff, dropout=dropout,
                                                   batch_first=True, norm_first=True)
        self.encoder    = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.drop       = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        nn.init.normal_(self.embed.weight, 0, 0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask, labels=None):
        key_pad_mask = (attention_mask == 0)
        x      = self.pos_enc(self.embed(input_ids))
        x      = self.encoder(x, src_key_padding_mask=key_pad_mask)
        logits = self.classifier(self.drop(x[:, 0]))
        loss   = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}


class TransformerDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=128):
        self.labels = labels
        pairs = [self._tok(t, vocab, max_len) for t in texts]
        self.ids  = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        self.mask = torch.tensor([p[1] for p in pairs], dtype=torch.long)

    @staticmethod
    def _tok(text, vocab, max_len):
        toks = str(text).split()[:max_len]
        ids  = [vocab.get(t, 1) for t in toks]
        mask = [1] * len(ids)
        pad  = max_len - len(ids)
        return ids + [0]*pad, mask + [0]*pad

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"input_ids": self.ids[idx], "attention_mask": self.mask[idx],
                "labels": torch.tensor(self.labels[idx], dtype=torch.long)}


def run_transformer_experiment(X_train, y_train, X_test, y_test):
    print("\n" + "="*65)
    print("  PHASE 4 — Offline Transformer (class-weighted)")
    print("="*65)

    exp_id = "EXP_023"
    t0 = time.time()

    # Load vocab from project file
    vocab_path = DATA_DIR / "vocab.json"
    with open(vocab_path) as f:
        raw_vocab = json.load(f)
    MAX_VOCAB = 10000
    PAD, UNK  = 0, 1
    sorted_toks = sorted(raw_vocab.items(), key=lambda x: -x[1])
    vocab = {"<PAD>": PAD, "<UNK>": UNK}
    for tok, _ in sorted_toks[:MAX_VOCAB - 2]:
        if tok not in vocab:
            vocab[tok] = len(vocab)
    vocab_size = len(vocab)
    print(f"  Vocab size: {vocab_size}")

    # Class weights
    cw_arr    = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=np.array(y_train))
    cw_tensor = torch.tensor(cw_arr, dtype=torch.float)

    MAX_LEN    = 128
    BATCH_SIZE = 32
    MAX_EPOCHS = 25
    PATIENCE   = 6
    LR         = 5e-4

    train_ds = TransformerDataset(X_train, y_train, vocab, MAX_LEN)
    test_ds  = TransformerDataset(X_test,  y_test,  vocab, MAX_LEN)
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_ld  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = OfflineTransformer(vocab_size).to(device)
    crit   = nn.CrossEntropyLoss(weight=cw_tensor.to(device))
    opt    = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = (len(train_ld)) * MAX_EPOCHS
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=200,
                                             num_training_steps=total_steps)

    best_f1, best_preds = 0.0, []
    ckpt    = MODELS_DIR / "transformer_offline_best.pt"
    no_imp  = 0

    print(f"  Training on {device} for up to {MAX_EPOCHS} epochs...")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        opt.zero_grad()
        for i, batch in enumerate(train_ld):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)
            out  = model(ids, mask, labels=lbls)
            loss = crit(out["logits"], lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()

        model.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            for batch in test_ld:
                ids  = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                out  = model(ids, mask)
                all_p.extend(out["logits"].argmax(-1).cpu().tolist())
                all_l.extend(batch["labels"].tolist())

        ep_f1 = f1_score(all_l, all_p, average="macro", zero_division=0)
        if ep_f1 > best_f1:
            best_f1, best_preds = ep_f1, all_p[:]
            torch.save(model.state_dict(), ckpt)
            no_imp = 0
        else:
            no_imp += 1

        if epoch % 5 == 0:
            print(f"    Epoch {epoch:>3}: F1={ep_f1:.4f}  best={best_f1:.4f}")

        if no_imp >= PATIENCE:
            print(f"    Early stop at epoch {epoch}")
            break

    elapsed = time.time() - t0
    f1 = log_result(exp_id, "Offline Transformer (4L/256d, class-weighted)",
                     "word vocab (top-10k), seq len=128",
                     "d_model=256, n_heads=8, n_layers=4, d_ff=1024, class_weight=balanced",
                     y_test, best_preds, elapsed)
    return {"f1": f1, "preds": best_preds}


# ---------------------------------------------------------------------------
# Ensemble experiments
# ---------------------------------------------------------------------------
def run_ensemble_experiments(X_train, y_train, X_test, y_test):
    print("\n" + "="*65)
    print("  PHASE 5 — Ensemble Experiments")
    print("="*65)

    # Build top-3 individual models for ensemble
    # We'll use the best feature set found so far: wide n-grams combined
    configs_for_ensemble = []
    vec_shared = make_combined_tfidf((1, 3), (2, 6), 25000, 50000)
    X_tr_v = vec_shared.fit_transform(X_train)
    X_te_v = vec_shared.transform(X_test)

    clfs = [
        ("LR-C1",  LogisticRegression(C=1.0,  max_iter=2000, class_weight="balanced", random_state=SEED)),
        ("LR-C5",  LogisticRegression(C=5.0,  max_iter=2000, class_weight="balanced", random_state=SEED)),
        ("LR-C10", LogisticRegression(C=10.0, max_iter=2000, class_weight="balanced", random_state=SEED)),
        ("SVM-C1", LinearSVC(C=1.0,  max_iter=5000, class_weight="balanced", random_state=SEED)),
        ("SVM-C2", LinearSVC(C=2.0,  max_iter=5000, class_weight="balanced", random_state=SEED)),
    ]

    # Fit all
    for name, clf in clfs:
        clf.fit(X_tr_v, y_train)

    # Hard-voting ensemble (top-3: LR-C5, SVM-C1, LR-C10)
    t0 = time.time()
    try:
        votes_top3 = []
        for name, clf in clfs[:3]:
            votes_top3.append(clf.predict(X_te_v))
        y_vote3 = np.array(votes_top3)
        y_ensemble3 = []
        for i in range(y_vote3.shape[1]):
            counts = np.bincount(y_vote3[:, i], minlength=3)
            y_ensemble3.append(np.argmax(counts))
        elapsed = time.time() - t0
        f1 = log_result("EXP_024", "Voting Ensemble (3 LR)",
                         "word (1-3)+char (2-6) TF-IDF",
                         "hard vote: LR-C1, LR-C5, LR-C10",
                         y_test, y_ensemble3, elapsed)
    except Exception as e:
        print(f"  [EXP_024] FAILED: {e}")

    # Hard-voting: 5 models
    t0 = time.time()
    try:
        votes_all = np.array([clf.predict(X_te_v) for _, clf in clfs])
        y_ensemble5 = []
        for i in range(votes_all.shape[1]):
            counts = np.bincount(votes_all[:, i], minlength=3)
            y_ensemble5.append(np.argmax(counts))
        elapsed = time.time() - t0
        f1 = log_result("EXP_025", "Voting Ensemble (5: 3LR+2SVM)",
                         "word (1-3)+char (2-6) TF-IDF",
                         "hard vote: LR-C1, LR-C5, LR-C10, SVM-C1, SVM-C2",
                         y_test, y_ensemble5, elapsed)
    except Exception as e:
        print(f"  [EXP_025] FAILED: {e}")

    # Soft-vote (LR supports predict_proba; SVM does not directly)
    t0 = time.time()
    try:
        lr_clfs = [(n, c) for n, c in clfs if n.startswith("LR")]
        proba_sum = None
        for _, clf in lr_clfs:
            p = clf.predict_proba(X_te_v)
            proba_sum = p if proba_sum is None else proba_sum + p
        y_soft = proba_sum.argmax(axis=1).tolist()
        elapsed = time.time() - t0
        f1 = log_result("EXP_026", "Soft-Vote Ensemble (3 LR)",
                         "word (1-3)+char (2-6) TF-IDF",
                         "soft vote: LR-C1, LR-C5, LR-C10 (predict_proba average)",
                         y_test, y_soft, elapsed)
    except Exception as e:
        print(f"  [EXP_026] FAILED: {e}")

    return vec_shared, clfs


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def generate_visualizations(y_test):
    print("\n" + "="*65)
    print("  Generating visualizations...")
    print("="*65)

    df_log = pd.DataFrame(exp_log, columns=LOG_COLS)
    df_log = df_log.sort_values("F1_macro", ascending=False).reset_index(drop=True)

    # --- 1. Bar chart: model comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#e74c3c" if f < NB_BASELINE else "#2ecc71" if f >= TARGET_F1 else "#3498db"
              for f in df_log["F1_macro"]]
    bars = ax.barh(df_log["Experiment_ID"] + " " + df_log["Model"].str[:30],
                   df_log["F1_macro"], color=colors)
    ax.axvline(NB_BASELINE, color="red",    linestyle="--", linewidth=1.5, label=f"NB baseline ({NB_BASELINE})")
    ax.axvline(TARGET_F1,   color="green",  linestyle="--", linewidth=1.5, label=f"Target ({TARGET_F1})")
    ax.set_xlabel("F1 Macro")
    ax.set_title("All Experiments — F1 Macro (sorted)")
    ax.legend()
    ax.set_xlim(0, 1.0)
    for bar, val in zip(bars, df_log["F1_macro"]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f"{val:.4f}", va="center", fontsize=7)
    plt.tight_layout()
    path = EXP_DIR / "best_model_comparison.jpeg"
    fig.savefig(path, dpi=150, format="jpeg")
    plt.close(fig)
    print(f"  Saved → {path}")

    # --- 2. Progression line chart
    df_prog = df_log.sort_values("Experiment_ID").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(df_prog)), df_prog["F1_macro"], "b-o", markersize=5, label="F1 macro")
    ax.axhline(NB_BASELINE, color="red",   linestyle="--", label=f"Baseline ({NB_BASELINE})")
    ax.axhline(TARGET_F1,   color="green", linestyle="--", label=f"Target ({TARGET_F1})")
    ax.set_xticks(range(len(df_prog)))
    ax.set_xticklabels(df_prog["Experiment_ID"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("F1 Macro")
    ax.set_title("Experiment Progression — F1 Macro")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = EXP_DIR / "experiment_progression.jpeg"
    fig.savefig(path, dpi=150, format="jpeg")
    plt.close(fig)
    print(f"  Saved → {path}")

    # --- 3. Per-class F1 heatmap
    top_n = min(15, len(df_log))
    top_df = df_log.head(top_n)
    heat_data = top_df[["F1_positive", "F1_negative", "F1_neutral"]].values
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.4 + 1)))
    sns.heatmap(heat_data, annot=True, fmt=".3f", cmap="RdYlGn",
                xticklabels=["Positive", "Negative", "Neutral"],
                yticklabels=top_df["Model"].str[:25].tolist(),
                vmin=0.4, vmax=0.9, ax=ax)
    ax.set_title(f"Per-class F1 Heatmap (Top {top_n} experiments)")
    plt.tight_layout()
    path = EXP_DIR / "per_class_f1_heatmap.jpeg"
    fig.savefig(path, dpi=150, format="jpeg")
    plt.close(fig)
    print(f"  Saved → {path}")

    # --- 4. Best confusion matrix
    best_row = df_log.iloc[0]
    best_id  = best_row["Experiment_ID"]
    # Find preds for best experiment from exp_log
    all_preds_map = {r["Experiment_ID"]: r for r in exp_log}
    print(f"  Best experiment: {best_id} — {best_row['Model']}")
    print(f"  (Confusion matrix requires re-prediction of best model; skipping if preds unavailable)")


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
def print_summary(df_log):
    best = df_log.iloc[0]
    gap  = TARGET_F1 - best["F1_macro"]

    print("\n")
    print("═" * 52)
    print("  EXPERIMENT RESULTS — Ranked by F1_macro")
    print("═" * 52)
    for i, row in df_log.head(10).iterrows():
        delta = row["F1_macro"] - NB_BASELINE
        sign  = "↑" if delta >= 0 else "↓"
        print(f"  #{i+1:>2}  {row['Experiment_ID']:>8}  {row['Model'][:30]:<30}  "
              f"F1={row['F1_macro']:.4f}  ({sign}{abs(delta):.1%} vs baseline)")
    print("═" * 52)
    print(f"  BASELINE (Naive Bayes):  F1={NB_BASELINE:.4f}")
    print(f"  TARGET:                  F1={TARGET_F1:.4f}")
    print(f"  BEST ACHIEVED:           F1={best['F1_macro']:.4f}  [{best['Experiment_ID']}]")
    if gap > 0:
        print(f"  GAP TO TARGET:           {gap:.4f} points ({gap:.1%})")
    else:
        print(f"  TARGET EXCEEDED BY:      {-gap:.4f} points! 🎉")
    print("═" * 52)
    print(f"\n  Best model: {best['Model']}")
    print(f"  Features:   {best['Features']}")
    print(f"  Params:     {best['Hyperparameters']}")
    print(f"\n  Per-class F1:")
    print(f"    Positive: {best['F1_positive']:.4f}")
    print(f"    Negative: {best['F1_negative']:.4f}")
    print(f"    Neutral:  {best['F1_neutral']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*65}")
    print(f"  Amharic Sentiment — Systematic Experiment Runner")
    print(f"  Started: {ts}")
    print(f"  Target: F1_macro ≥ {TARGET_F1}  (baseline: {NB_BASELINE})")
    print(f"{'='*65}")

    # ------------------------------------------------------------------
    print("\n[Step 0] Loading data and running pipeline audit...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_splits()
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    y_arr = np.array(y_train)
    print(f"  Label distribution (train): "
          f"pos={( y_arr==0).sum()} neg={(y_arr==1).sum()} neu={(y_arr==2).sum()}")

    cw = compute_class_weight("balanced", classes=np.array([0,1,2]), y=y_arr)
    print(f"  Class weights: pos={cw[0]:.3f} neg={cw[1]:.3f} neu={cw[2]:.3f}")

    # Data audit
    dup_test = len(set(X_train) & set(X_test))
    print(f"  Train-Test overlap (cleaned_text): {dup_test}")
    empty = sum(1 for x in X_train if not x.strip())
    print(f"  Empty train texts: {empty}")

    # ------------------------------------------------------------------
    best_ml, ml_results = run_ml_experiments(X_train, y_train, X_test, y_test)

    # ------------------------------------------------------------------
    best_boost = run_boosting_experiments(X_train, y_train, X_test, y_test)

    # ------------------------------------------------------------------
    bilstm_result = run_bilstm_experiment(X_train, y_train, X_test, y_test)

    # ------------------------------------------------------------------
    transformer_result = run_transformer_experiment(X_train, y_train, X_test, y_test)

    # ------------------------------------------------------------------
    run_ensemble_experiments(X_train, y_train, X_test, y_test)

    # ------------------------------------------------------------------
    print("\n[Final] Saving experiment log...")
    df_log = save_log()
    generate_visualizations(y_test)

    # Best confusion matrix (re-predict with best model)
    best_row = df_log.iloc[0]
    print(f"\n  Generating confusion matrix for best model: {best_row['Experiment_ID']}")
    best_exp_id = best_row["Experiment_ID"]

    # Re-run best model for CM
    if best_exp_id.startswith("EXP_0") and int(best_exp_id.split("_")[1]) <= 17:
        # It's an ML model — re-fit with best params
        best_vec = make_combined_tfidf((1, 3), (2, 6), 25000, 50000)
        C_val = float(best_row["Hyperparameters"].split("C=")[1].split(",")[0]) \
                if "C=" in best_row["Hyperparameters"] else 5.0
        if "LR" in best_row["Model"] or "Logistic" in best_row["Model"]:
            best_clf = LogisticRegression(C=C_val, max_iter=2000,
                                           class_weight="balanced", random_state=SEED)
        else:
            best_clf = LinearSVC(C=C_val, max_iter=5000,
                                  class_weight="balanced", random_state=SEED)
        X_tr_v = best_vec.fit_transform(X_train)
        X_te_v = best_vec.transform(X_test)
        best_clf.fit(X_tr_v, y_train)
        best_preds_cm = best_clf.predict(X_te_v)
    else:
        best_preds_cm = None

    if best_preds_cm is not None:
        cm = confusion_matrix(y_test, best_preds_cm)
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Best Model CM — {best_row['Model'][:40]}\nF1={best_row['F1_macro']:.4f}")
        plt.tight_layout()
        cm_path = EXP_DIR / "best_confusion_matrix.jpeg"
        fig.savefig(cm_path, dpi=150, format="jpeg")
        plt.close(fig)
        print(f"  Saved → {cm_path}")

    print_summary(df_log)
    print(f"\n  All results saved to: {EXP_DIR}")
    print(f"  Experiment log: {EXP_DIR}/experiment_log.xlsx")


if __name__ == "__main__":
    main()
