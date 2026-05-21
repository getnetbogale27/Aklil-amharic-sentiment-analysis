"""
Transformer fine-tuning for Amharic sentiment analysis — CPU-optimized.
Supports multiple model variants via command-line args.
Data: data/raw/amh/{train,dev,test}.tsv  (columns: tweet, label)
"""

import argparse
import json
import os
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from datetime import datetime
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset

# ── Argument parser ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--max_length", type=int, default=128)
parser.add_argument("--gradient_accumulation", type=int, default=4)
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--output_dir", type=str, default="results/transformer_runs")
parser.add_argument("--experiment_name", type=str, default=None)
parser.add_argument("--freeze_layers", type=int, default=0,
                    help="Freeze all but last N transformer layers + classifier/pooler")
args = parser.parse_args()

# ── Setup ──────────────────────────────────────────────────────────────────────
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
exp_name = args.experiment_name or args.model_name.split("/")[-1]
RUN_DIR = os.path.join(args.output_dir, f"{exp_name}_{TIMESTAMP}")
os.makedirs(RUN_DIR, exist_ok=True)

print(f"\n{'='*60}")
print(f"  Transformer Training — CPU Mode")
print(f"  Model: {args.model_name}")
print(f"  Output: {RUN_DIR}")
print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}\n")

# ── Load Data ──────────────────────────────────────────────────────────────────
DATA_DIR = "data/raw/amh"

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.tsv"), sep="\t")
dev_df   = pd.read_csv(os.path.join(DATA_DIR, "dev.tsv"),   sep="\t")
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.tsv"),  sep="\t")

text_col  = "tweet"
label_col = "label"

# Consistent alphabetical label ordering (matches sklearn's sort in class_weight)
label_map     = {l: i for i, l in enumerate(sorted(train_df[label_col].unique()))}
inv_label_map = {v: k for k, v in label_map.items()}
print(f"  Labels: {label_map}")
print(f"  Train: {len(train_df)} | Val: {len(dev_df)} | Test: {len(test_df)}")

y_train = train_df[label_col].map(label_map).values
class_weights = compute_class_weight(
    "balanced", classes=np.unique(y_train), y=y_train
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
print(f"  Class weights: {dict(zip(sorted(label_map.keys()), class_weights.round(3)))}")

# ── Tokenize ───────────────────────────────────────────────────────────────────
print(f"\n  Loading tokenizer: {args.model_name}")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

def tokenize_data(df):
    texts  = df[text_col].astype(str).tolist()
    labels = df[label_col].map(label_map).tolist()
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=args.max_length,
    )
    return Dataset.from_dict({
        "input_ids":      encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels":         labels,
    })

print("  Tokenizing datasets...")
train_dataset = tokenize_data(train_df)
dev_dataset   = tokenize_data(dev_df)
test_dataset  = tokenize_data(test_df)

# ── Model ──────────────────────────────────────────────────────────────────────
print(f"  Loading model: {args.model_name}")
model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,
    num_labels=len(label_map),
)

# Optional layer freezing
if args.freeze_layers > 0:
    num_hidden = model.config.num_hidden_layers
    freeze_up_to = num_hidden - args.freeze_layers
    for name, param in model.named_parameters():
        if "classifier" in name or "pooler" in name:
            continue  # always train head
        layer_num = None
        for part in name.split("."):
            if part.isdigit():
                layer_num = int(part)
        if layer_num is not None and layer_num < freeze_up_to:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Frozen: {total - trainable:,} / {total:,} params. Training: {trainable:,}")
else:
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total:,}")

param_count = sum(p.numel() for p in model.parameters())

# ── Custom Trainer with class-weighted loss ────────────────────────────────────
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss    = nn.CrossEntropyLoss(
            weight=class_weights_tensor.to(logits.device)
        )(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":        accuracy_score(labels, preds),
        "f1_macro":        f1_score(labels, preds, average="macro"),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro":    recall_score(labels, preds, average="macro", zero_division=0),
    }

# ── Training Arguments (CPU-optimized) ────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=RUN_DIR,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size * 2,
    gradient_accumulation_steps=args.gradient_accumulation,
    learning_rate=args.lr,
    weight_decay=args.weight_decay,
    warmup_ratio=args.warmup_ratio,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    save_total_limit=2,
    logging_steps=50,
    report_to="none",
    use_cpu=True,
    dataloader_num_workers=0,
    fp16=False,
)

# ── Train ──────────────────────────────────────────────────────────────────────
print(f"\n  Training config:")
print(f"    Epochs:               {args.epochs}")
print(f"    Batch size:           {args.batch_size}")
print(f"    Gradient accumulation:{args.gradient_accumulation}")
print(f"    Effective batch size: {args.batch_size * args.gradient_accumulation}")
print(f"    Learning rate:        {args.lr}")
print(f"    Max sequence length:  {args.max_length}")
print(f"\n  Starting training...")

t0 = time.time()

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

train_result = trainer.train()
train_time   = time.time() - t0
print(f"\n  Training complete in {train_time/60:.1f} minutes")

# ── Evaluate on Test Set ───────────────────────────────────────────────────────
print("\n  Evaluating on test set...")
test_results = trainer.predict(test_dataset)
test_preds   = np.argmax(test_results.predictions, axis=-1)
test_labels  = np.array(test_dataset["labels"])

test_acc  = accuracy_score(test_labels, test_preds)
test_f1   = f1_score(test_labels, test_preds, average="macro")
test_prec = precision_score(test_labels, test_preds, average="macro", zero_division=0)
test_rec  = recall_score(test_labels, test_preds, average="macro", zero_division=0)
test_f1w  = f1_score(test_labels, test_preds, average="weighted")

print(f"\n  ╔══════════════════════════════════════╗")
print(f"  ║  TEST RESULTS                        ║")
print(f"  ║  Accuracy:          {test_acc:.4f}             ║")
print(f"  ║  F1 (macro):        {test_f1:.4f}             ║")
print(f"  ║  Precision (macro): {test_prec:.4f}             ║")
print(f"  ║  Recall (macro):    {test_rec:.4f}             ║")
print(f"  ╚══════════════════════════════════════╝")

sorted_labels = sorted(label_map.keys())
report = classification_report(
    test_labels, test_preds,
    target_names=sorted_labels,
    output_dict=True,
    zero_division=0,
)
print(f"\n  Per-class results:")
for cls_name in sorted_labels:
    r = report[cls_name]
    print(f"    {cls_name:12s}  P={r['precision']:.4f}  R={r['recall']:.4f}  "
          f"F1={r['f1-score']:.4f}  n={int(r['support'])}")

# ── Save Results JSON ──────────────────────────────────────────────────────────
steps_per_epoch = len(train_dataset) / (args.batch_size * args.gradient_accumulation)
epochs_trained  = int(train_result.global_step / max(steps_per_epoch, 1))

results = {
    "model_name":       args.model_name,
    "experiment_name":  exp_name,
    "timestamp":        TIMESTAMP,
    "parameters":       param_count,
    "train_time_minutes": round(train_time / 60, 1),
    "epochs_trained":   epochs_trained,
    "hyperparameters": {
        "learning_rate":        args.lr,
        "batch_size":           args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": args.batch_size * args.gradient_accumulation,
        "max_length":           args.max_length,
        "warmup_ratio":         args.warmup_ratio,
        "weight_decay":         args.weight_decay,
        "freeze_layers":        args.freeze_layers,
    },
    "test_results": {
        "accuracy":        round(test_acc,  4),
        "f1_macro":        round(test_f1,   4),
        "f1_weighted":     round(test_f1w,  4),
        "precision_macro": round(test_prec, 4),
        "recall_macro":    round(test_rec,  4),
    },
    "per_class": {
        cls_name: {k: round(v, 4) for k, v in report[cls_name].items()}
        for cls_name in sorted_labels
    },
    "class_weights":   {k: round(v, 4) for k, v in zip(sorted_labels, class_weights)},
    "confusion_matrix": confusion_matrix(test_labels, test_preds).tolist(),
    "train_history":    trainer.state.log_history,
}

with open(os.path.join(RUN_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

# ── Save Best Model ────────────────────────────────────────────────────────────
model_save_path = os.path.join(RUN_DIR, "best_model")
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"\n  Model saved: {model_save_path}")

# ── Save Predictions ───────────────────────────────────────────────────────────
pred_df = test_df.copy()
pred_df["predicted"] = [inv_label_map[p] for p in test_preds]
pred_df["correct"]   = pred_df[label_col] == pred_df["predicted"]
pred_df.to_csv(os.path.join(RUN_DIR, "test_predictions.csv"), index=False)

# ── Plots ──────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion matrix
cm  = confusion_matrix(test_labels, test_preds)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=sorted_labels, yticklabels=sorted_labels)
ax.set_title(f"Confusion Matrix — {exp_name}\nF1_macro={test_f1:.4f}",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
fig.savefig(os.path.join(RUN_DIR, "confusion_matrix.jpeg"), dpi=150, format="jpeg")
plt.close()

# Training curves
history      = trainer.state.log_history
eval_entries = [h for h in history if "eval_loss" in h]
eval_losses  = [h["eval_loss"] for h in eval_entries]
eval_f1s     = [h.get("eval_f1_macro", 0) for h in eval_entries]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
if eval_losses:
    axes[0].plot(range(1, len(eval_losses) + 1), eval_losses, "r-o", label="Val Loss")
    axes[0].set_title("Validation Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

if eval_f1s:
    axes[1].plot(range(1, len(eval_f1s) + 1), eval_f1s, "g-o", label="Val F1 (macro)")
    axes[1].axhline(y=0.6802, color="orange", linestyle="--",
                    label="MNB baseline (0.6802)", linewidth=1.5)
    axes[1].axhline(y=0.5776, color="red", linestyle=":",
                    label="Bi-LSTM (0.5776)", linewidth=1.2)
    axes[1].set_title("Validation F1 Score", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1 (macro)")
    axes[1].legend()

plt.suptitle(f"{exp_name} Training Curves", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(RUN_DIR, "training_curves.jpeg"), dpi=150, format="jpeg")
plt.close()

print(f"\n  All outputs saved to: {RUN_DIR}/")
print(f"  Done at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\n{'='*60}")
print(f"  SUMMARY: {exp_name}")
print(f"  F1_macro = {test_f1:.4f}  |  Accuracy = {test_acc:.4f}")
print(f"  vs MNB baseline 0.6802: {test_f1 - 0.6802:+.4f}")
print(f"{'='*60}\n")
