# Developing Deep Learning-Based Sentiment Analysis of Amharic Social Media for Public Policy Enhancement in Ethiopia

> **Academic Research Project** · Addis Ababa University  
> Author: **Getnet Bogale**

---

## Overview

This repository contains the full implementation for a research project that applies **deep learning** to classify the sentiment of Amharic-language social media posts, with the goal of informing public policy decisions in Ethiopia.

Amharic is the official working language of the Ethiopian federal government and is spoken by over 30 million people. Despite its importance, Amharic remains severely under-resourced in the NLP community. This project addresses that gap by:

1. Leveraging the **AfriSenti** benchmark dataset for Amharic Twitter sentiment
2. Augmenting it with **additionally collected policy-related Amharic tweets**
3. Training and comparing deep learning models (Bi-LSTM, XLM-RoBERTa)
4. Deploying a **policy sentiment dashboard** for decision-makers

---

## Research Objectives

| # | Objective |
|---|-----------|
| 1 | Collect and annotate a domain-specific Amharic corpus of public-policy tweets |
| 2 | Develop an Amharic-specific text preprocessing pipeline (normalisation, tokenisation, stopword removal) |
| 3 | Train and evaluate CNN, Bi-LSTM, and transformer-based sentiment classifiers |
| 4 | Identify which deep learning architecture achieves the best macro-F1 on Amharic sentiment |
| 5 | Build an interactive policy dashboard that visualises public sentiment trends for government stakeholders |

---

## Datasets

### AfriSenti — Amharic (`data/raw/amh/`)

| Split | File | Size |
|-------|------|------|
| Train | `train.tsv` | ~8,000 tweets |
| Dev   | `dev.tsv`   | ~1,000 tweets |
| Test  | `test.tsv`  | ~1,000 tweets |

Tab-separated format: `tweet<TAB>label` where label ∈ {`positive`, `negative`, `neutral`}.

AfriSenti also includes 13 other African languages used in transfer-learning experiments (`data/raw/arq`, `ary`, `hau`, `ibo`, `kin`, `orm`, `pcm`, `por`, `swa`, `tir`, `tso`, `twi`, `yor`).

### Additionally Collected Policy Tweets (`data/raw/dataset.xlsx`)

Amharic tweets collected from Twitter/X using keywords related to Ethiopian government policies (education, health, economy, security). Annotated with the same three-class scheme. See `docs/` for annotation guidelines.

---

## Folder Structure

```
Aklil-amharic-sentiment-analysis/
│
├── data/
│   ├── raw/                    ← original, unmodified data
│   │   ├── amh/                ← AfriSenti Amharic (train/dev/test.tsv)
│   │   ├── arq/ … yor/         ← other AfriSenti languages
│   │   └── dataset.xlsx        ← additionally collected policy tweets
│   ├── processed/              ← output of preprocessing pipeline
│   └── annotated/              ← final annotated/merged datasets
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── src/
│   ├── preprocessing/
│   │   └── preprocess.py       ← Amharic normalisation, tokenisation, stopwords
│   ├── models/
│   │   ├── bilstm_model.py     ← Bi-LSTM + self-attention (PyTorch)
│   │   └── transformer_model.py← XLM-RoBERTa fine-tuning (HuggingFace)
│   ├── training/
│   │   └── train.py            ← training loop, checkpointing, LR scheduling
│   ├── evaluation/
│   │   └── evaluate.py         ← accuracy, precision, recall, F1, confusion matrix
│   └── dashboard/              ← Flask policy-sentiment dashboard
│
├── results/
│   ├── figures/                ← confusion matrices, training curves, charts
│   ├── metrics/                ← JSON/CSV evaluation results per run
│   └── models/                 ← saved model checkpoints (.pt)
│
├── configs/
│   └── model_config.yaml       ← all hyperparameters in one place
│
├── tests/                      ← unit tests (pytest)
├── docs/                       ← report drafts, annotation guidelines
├── requirements.txt
└── .gitignore
```

---

## Setup

### 1 · Clone the repository

```bash
git clone https://github.com/getnetbogale27/Aklil-amharic-sentiment-analysis.git
cd Aklil-amharic-sentiment-analysis
```

### 2 · Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows
```

### 3 · Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **GPU users:** Install PyTorch with CUDA support first — see [pytorch.org](https://pytorch.org/get-started/locally/) — then run `pip install -r requirements.txt`.

### 4 · (Optional) Download NLTK data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

---

## How to Run

### Preprocessing

Converts raw AfriSenti TSV files into cleaned, normalised, tokenised files:

```bash
python -m src.preprocessing.preprocess \
    --input_dir  data/raw/amh \
    --output_dir data/processed/amh
```

Output files: `data/processed/amh/{train,dev,test}_processed.tsv`

### Training

**XLM-RoBERTa (recommended — best accuracy):**

```bash
python -m src.training.train \
    --model transformer \
    --config configs/model_config.yaml \
    --data_dir data/processed/amh \
    --output_dir results/models
```

**Bi-LSTM:**

```bash
python -m src.training.train \
    --model bilstm \
    --config configs/model_config.yaml
```

Training logs and the best checkpoint are saved to `results/models/`.

### Evaluation

```bash
python -m src.evaluation.evaluate \
    --checkpoint results/models/transformer_best.pt \
    --model transformer \
    --config configs/model_config.yaml \
    --data_dir data/processed/amh \
    --run_name xlmr_test
```

Results (JSON + CSV) → `results/metrics/`  
Confusion matrix figure → `results/figures/`

### Data Exploration Notebook

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

---

## Models

| Model | Architecture | Pre-training | Expected Macro-F1 |
|-------|-------------|-------------|-------------------|
| Bi-LSTM | 2-layer Bi-LSTM + self-attention | None (trained from scratch) | ~0.65 |
| XLM-RoBERTa-base | 12-layer transformer | 100 languages (CC-100) | ~0.72 |
| XLM-RoBERTa-large | 24-layer transformer | 100 languages (CC-100) | ~0.75 |

*Estimates based on AfriSenti shared-task results; actual results depend on your training setup.*

---

## Configuration

All hyperparameters live in `configs/model_config.yaml`. Key options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `transformer_checkpoint` | `xlm-roberta-base` | HuggingFace model ID |
| `batch_size` | `32` | Training batch size |
| `num_epochs` | `10` | Maximum training epochs |
| `learning_rate` | `2e-5` | AdamW learning rate |
| `max_length` | `128` | Maximum sub-word tokens per tweet |
| `num_classes` | `3` | pos / neg / neu |

---

## Citation

If you use this codebase or the AfriSenti dataset in your research, please cite:

```bibtex
@inproceedings{muhammad-etal-2023-afrisenti,
  title     = {{AfriSenti}: A Twitter Sentiment Analysis Benchmark for
               African Languages},
  author    = {Muhammad, Shamsuddeen Hassan and
               Abdulmumin, Idris and
               Ayele, Abinew Ali and
               others},
  booktitle = {Proceedings of EMNLP 2023},
  year      = {2023},
  url       = {https://arxiv.org/abs/2302.08956}
}

@article{conneau-etal-2020-xlmr,
  title   = {Unsupervised Cross-lingual Representation Learning at Scale},
  author  = {Conneau, Alexis and Khandelwal, Kartikay and Goyal, Naman
             and others},
  journal = {Proceedings of ACL 2020},
  year    = {2020},
  url     = {https://arxiv.org/abs/1911.02116}
}

@inproceedings{ayele-etal-2023-amharic,
  title     = {Exploring Amharic Sentiment Analysis from Social Media Texts:
               Building Annotation Tools and Classification Models},
  author    = {Ayele, Abinew Ali and Yimam, Seid Muhie and
               Belay, Tadesse Destaw and others},
  booktitle = {Proceedings of RANLP 2023},
  year      = {2023}
}
```

---

## License

The code in this repository is released under the **MIT License**.  
The AfriSenti dataset is distributed under its own license — see `data/raw/README.txt`.

---

## Contact

**Getnet Bogale** — PhD Researcher, Addis Ababa University  
GitHub: [@getnetbogale27](https://github.com/getnetbogale27)
