"""
Training script for Amharic sentiment analysis models.

Supports both the Bi-LSTM and XLM-RoBERTa models via a --model flag.
Configuration is loaded from configs/model_config.yaml.

Usage:
    python -m src.training.train --model bilstm --config configs/model_config.yaml
    python -m src.training.train --model transformer --config configs/model_config.yaml
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.evaluation.evaluate import compute_metrics
from src.models.bilstm_model import build_bilstm
from src.models.transformer_model import build_transformer, load_tokenizer, tokenize_batch
from src.preprocessing.preprocess import preprocess_dataframe

import pandas as pd


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset wrappers
# ---------------------------------------------------------------------------
LABEL_MAP = {"positive": 0, "negative": 1, "neutral": 2}


class AmharicDataset(Dataset):
    """Generic dataset for tokenised Amharic sentiment data."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 128):
        self.encodings = tokenize_batch(texts, tokenizer, max_length)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def load_split(path: Path) -> tuple[list[str], list[int]]:
    """Load a preprocessed TSV split into (texts, labels)."""
    df = pd.read_csv(path, sep="\t")
    texts = df["cleaned_text"].tolist()
    labels = [LABEL_MAP[l] for l in df["label"].tolist()]
    return texts, labels


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, device, model_type: str) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="  train", leave=False):
        optimizer.zero_grad()
        if model_type == "transformer":
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out["loss"]
        else:  # bilstm
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids)
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_epoch(model, loader, device, model_type: str) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  eval", leave=False):
            if model_type == "transformer":
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out["logits"]
            else:
                input_ids = batch["input_ids"].to(device)
                logits = model(input_ids)
            preds = logits.argmax(dim=-1).cpu().tolist()
            labels = batch["labels"].tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)
    return compute_metrics(all_labels, all_preds)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bilstm", "transformer"], default="transformer")
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--data_dir", default="data/processed/amh")
    parser.add_argument("--output_dir", default="results/models")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Model: {args.model}")

    data_dir = Path(args.data_dir)
    train_texts, train_labels = load_split(data_dir / "train_processed.tsv")
    dev_texts,   dev_labels   = load_split(data_dir / "dev_processed.tsv")

    tokenizer = load_tokenizer(config.get("transformer_checkpoint", "xlm-roberta-base"))

    train_ds = AmharicDataset(train_texts, train_labels, tokenizer, config.get("max_length", 128))
    dev_ds   = AmharicDataset(dev_texts,   dev_labels,   tokenizer, config.get("max_length", 128))

    batch_size = config.get("batch_size", 32)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=batch_size)

    model = (build_transformer(config) if args.model == "transformer" else build_bilstm(config))
    model = model.to(device)

    lr = float(config.get("learning_rate", 2e-5))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    num_epochs = config.get("num_epochs", 10)
    history = []

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device, args.model)
        val_metrics = evaluate_epoch(model, dev_loader, device, args.model)
        scheduler.step(val_metrics["macro_f1"])
        print(f"  loss={train_loss:.4f}  val_f1={val_metrics['macro_f1']:.4f}")
        history.append({"epoch": epoch, "loss": train_loss, **val_metrics})

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            ckpt = output_dir / f"{args.model}_best.pt"
            torch.save(model.state_dict(), ckpt)
            print(f"  Saved best checkpoint → {ckpt}")

    with open(output_dir / f"{args.model}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. Best macro-F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
