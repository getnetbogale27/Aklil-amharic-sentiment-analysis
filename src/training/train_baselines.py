"""Train classical ML baselines for Amharic sentiment analysis."""

import os
import pickle
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import joblib

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "results", "models")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")
METRICS_DIR = os.path.join(BASE_DIR, "results", "metrics")

for d in (MODELS_DIR, FIGURES_DIR, METRICS_DIR):
    os.makedirs(d, exist_ok=True)

TEXT_COL = "cleaned_text"
LABEL_COL = "label"
LABEL_NAMES = ["positive", "negative", "neutral"]


def load_split(name):
    path = os.path.join(DATA_DIR, f"{name}.csv")
    df = pd.read_csv(path).dropna(subset=[TEXT_COL, LABEL_COL])
    return df[TEXT_COL].astype(str).tolist(), df[LABEL_COL].astype(int).tolist()


def plot_confusion_matrix(cm, model_name, labels):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix – {model_name}")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"cm_{model_name.lower().replace(' ', '_')}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved confusion matrix → {path}")


def evaluate(model, X, y_true, model_name, split_name):
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    p_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    r_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n{'='*60}")
    print(f"  {model_name}  |  {split_name.upper()}")
    print(f"{'='*60}")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Precision (mac) : {p_macro:.4f}")
    print(f"  Recall (mac)    : {r_macro:.4f}")
    print(f"  F1 macro        : {f1_macro:.4f}")
    print(f"  F1 weighted     : {f1_weighted:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, zero_division=0))

    if split_name == "test":
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, model_name, LABEL_NAMES)

    return {
        "Model": model_name,
        "Split": split_name,
        "Accuracy": round(acc, 4),
        "Precision": round(p_macro, 4),
        "Recall": round(r_macro, 4),
        "F1_macro": round(f1_macro, 4),
        "F1_weighted": round(f1_weighted, 4),
    }


def main():
    print("Loading data …")
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    print("Fitting TF-IDF vectorizer …")
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)

    tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    joblib.dump(tfidf, tfidf_path)
    print(f"Saved TF-IDF vectorizer → {tfidf_path}")

    models = [
        ("Naive Bayes",          MultinomialNB(),                           "nb_model.pkl"),
        ("Logistic Regression",  LogisticRegression(max_iter=1000, C=1.0),  "lr_model.pkl"),
        ("SVM",                  LinearSVC(max_iter=2000),                  "svm_model.pkl"),
        ("KNN",                  KNeighborsClassifier(n_neighbors=5),       "knn_model.pkl"),
    ]

    all_rows = []

    for model_name, clf, pkl_name in models:
        print(f"\nTraining {model_name} …")
        clf.fit(X_train_tfidf, y_train)

        model_path = os.path.join(MODELS_DIR, pkl_name)
        joblib.dump(clf, model_path)
        print(f"  Saved → {model_path}")

        val_row = evaluate(clf, X_val_tfidf, y_val, model_name, "val")
        test_row = evaluate(clf, X_test_tfidf, y_test, model_name, "test")
        all_rows.extend([val_row, test_row])

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(os.path.join(METRICS_DIR, "all_results.csv"), index=False)

    # Comparison table (test split only), sorted by F1_macro descending
    test_df = (
        results_df[results_df["Split"] == "test"]
        .drop(columns="Split")
        .sort_values("F1_macro", ascending=False)
        .reset_index(drop=True)
    )
    comparison_path = os.path.join(METRICS_DIR, "baseline_comparison.csv")
    test_df.to_csv(comparison_path, index=False)
    print(f"\nSaved comparison table → {comparison_path}")

    print("\n" + "=" * 60)
    print("  FINAL COMPARISON (test set, sorted by F1_macro)")
    print("=" * 60)
    print(test_df.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
