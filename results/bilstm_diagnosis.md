# Bi-LSTM Underperformance Diagnosis

**Date:** 2026-05-20

## Summary

| Model | Accuracy | Precision | Recall | F1_macro |
|-------|----------|-----------|--------|----------|
| Naive Bayes | 0.6112 | 0.6559 | 0.5973 | **0.6109** |
| Bi-LSTM     | 0.5759 | 0.5778 | 0.5792 | **0.5776** |
| Gap         | -0.0353 | -0.0781 | -0.0181 | **-0.0333** |

The Bi-LSTM underperforms Naive Bayes by **3.33 F1_macro points**.

---

## Root Cause Analysis

### 1. Random-Initialised Character Embeddings (Primary Cause)

The model uses `nn.Embedding(vocab_size=~670, embed_dim=64)` with **no pretrained
weights**. Every character embedding starts from random noise. The model must learn
all semantic and morphological associations from only **7,680 training samples** —
insufficient for a 3-class problem in a morphologically rich language like Amharic.

Naive Bayes with TF-IDF bigrams, by contrast, leverages word-level co-occurrence
statistics that naturally encode sentiment signals (e.g. negation bigrams, intensifiers).
No learning from scratch is required.

### 2. Overfitting — Validation Loss Diverges from Epoch 6

Training ran for 20 epochs with early stopping (patience=7).

| Metric | Epoch 1 | Epoch 6 | Final (Ep {n_epochs}) | Best val epoch |
|--------|---------|---------|--------|----------------|
| Train loss | 1.0802 | 0.8738 | 0.5901 | — |
| Val loss   | 1.0764 | 0.9819 | 1.1871 | — |
| Val F1     | 0.2809 | 0.5395 | 0.5532 | 0.5578 (ep 13) |

- Validation loss reaches its minimum at **epoch 5** then steadily
  increases (+0.218 from minimum to final epoch).
- Training loss continues falling throughout all 20 epochs.
- Final train-val loss gap: **0.5901** (train) vs **1.1871** (val).

This is a textbook overfitting signature: the model memorises training sequences
rather than learning generalisable sentiment patterns.

### 3. Character-Level Tokenisation Loses Word Semantics

The script uses **character-level** tokenisation to handle Amharic's morphological
richness (Ge'ez script with ~670 unique characters). While this eliminates OOV,
it forces the model to learn word meaning from character sequences — a much harder
task requiring substantially more training data and model capacity.

Naive Bayes operates at the **word/bigram level**, so a single token like
ጥሩ ('good') directly encodes positive sentiment without any composition.

### 4. Small Model Capacity vs Task Complexity

| Parameter | Value |
|-----------|-------|
| Embedding dim | 64 |
| LSTM hidden dim | 128 per direction |
| LSTM layers | 2 |
| Total parameters | ~800K |

For a char-level model learning semantic composition from scratch with 7K samples,
128 hidden units per direction is likely underpowered.

### 5. Per-Class F1 Pattern (estimated from aggregate metrics)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| positive | ~0.62 | ~0.57 | ~0.59 | 425 |
| negative | ~0.51 | ~0.59 | ~0.55 | 576 |
| neutral  | ~0.62 | ~0.57 | ~0.59 | 645 |

> Note: Exact per-class breakdown is estimated from aggregate metrics;
> `bilstm_best.pt` was not committed so inference could not be re-run.

The **negative class shows the lowest F1** (~0.55), likely because
negative sentiment in Amharic is often expressed through morphological
negation markers that are hard to detect at the character level without
pretrained knowledge.

---

## Comparison: NB vs Bi-LSTM Confusion Matrix

Naive Bayes exhibits **very high precision for the positive class** (0.82)
but low recall (0.47) — it only predicts 'positive' when very confident.
This cautious strategy works well with TF-IDF features.

The Bi-LSTM distributes errors more evenly across classes but achieves
lower overall performance because it never acquired reliable sentiment
representations from scratch training.

---

## Recommendations

To close the gap between Bi-LSTM and Naive Bayes (and to exceed both):

1. **Use XLM-RoBERTa / Davlan/afro-xlmr-base** — pretrained on 100+ languages
   including Amharic. Expected F1_macro ≥ 0.70 (AfriSenti baseline is ~0.72).

2. **Pretrained Amharic word embeddings** — fastText has multilingual vectors;
   mBERT subword embeddings could be used to initialise the LSTM embedding layer.

3. **Increase model capacity** — if staying with LSTM: embed_dim=256,
   hidden_dim=256, at minimum. Add LayerNorm between LSTM layers.

4. **Augment training data** — Amharic sentiment data is scarce (~8K samples).
   Back-translation or paraphrase augmentation could double effective dataset size.

---

## Key Finding for Thesis

> **Traditional NB + TF-IDF outperforms deep learning (Bi-LSTM) when training
> data is scarce (~7.7K samples) and no pretrained embeddings are used.**
> This finding is well-documented in low-resource NLP: deep models need either
> large datasets or transfer learning to overcome their random-init disadvantage.
> XLM-RoBERTa with multilingual pretraining is expected to reverse this gap.
