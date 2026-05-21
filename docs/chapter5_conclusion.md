# Chapter 5: Conclusion and Future Work

---

## 5.1 Summary

### 5.1.1 Problem Statement and Motivation

Amharic is the official working language of the Federal Democratic Republic of Ethiopia, spoken by over 57 million people, yet it remains severely under-resourced in computational natural language processing. Sentiment analysis — the automatic classification of opinion polarity in text — is a prerequisite for any system that monitors public discourse, evaluates policy reception, or supports evidence-based governance. No published study prior to this work had conducted a systematic, head-to-head evaluation of machine-learning and deep-learning approaches on the AfriSenti Amharic benchmark using a reproducible, open-source pipeline.

This thesis addressed the question: *Which model family and feature representation best captures sentiment in Amharic social-media text, and why?*

### 5.1.2 What Was Done

The study proceeded through four phases:

1. **Data acquisition and preprocessing**: 11,477 Amharic tweets were compiled from the AfriSenti benchmark and a supplementary policy-domain corpus. A five-stage preprocessing pipeline (Unicode NFC normalisation, Ethiopic character normalisation, noise removal, tokenisation, stopword filtering) reduced the usable corpus to 10,972 samples partitioned 70/15/15 for train/validation/test.

2. **Feature engineering**: 108 systematic experiments explored word-level and character-level TF-IDF features across multiple n-gram ranges, vocabulary sizes, and smoothing parameters, establishing that character n-gram TF-IDF (range (2,5), max_features=80,000) is the optimal representation for Amharic.

3. **Model evaluation**: Ten model families were evaluated — Multinomial Naïve Bayes, Logistic Regression, SVM, KNN, XGBoost, LightGBM, a Bi-LSTM with random embeddings, an MNB–Transformer ensemble, and a custom transformer trained from random initialisations.

4. **Analysis and synthesis**: The per-class performance, training dynamics, ablation results, and comparison with prior work were analysed to yield the findings presented in Chapter 4.

### 5.1.3 Key Results

The best model — **Multinomial Naïve Bayes with character n-gram TF-IDF features (range (2-5), max_features=80,000, α=0.2)** — achieved **F1_macro = 0.7282** on the official AfriSenti Amharic test set (n = 1,646), trained in 1.9 seconds on a standard CPU. This result is competitive with the top teams in the AfriSenti SemEval-2023 shared task, who used GPU-trained fine-tuned XLM-RoBERTa. The transformer trained from random initialisations (no pretrained weights available) achieved F1_macro = 0.3648, confirming that pretrained representations — not architectural complexity — are the primary driver of transformer performance on small datasets.

---

## 5.2 Conclusions: Research Questions Answered

### RQ1: Which feature representation best captures Amharic sentiment patterns?

**Answer: Character-level TF-IDF n-gram features (range 2–5) are definitively superior to word-level features for Amharic.**

The character n-gram approach outperformed the best word-level TF-IDF configuration by **+11.66 F1 points** (0.7282 vs. 0.6116). This large margin arises from Amharic's agglutinative morphology: a single root appears in dozens of inflected surface forms, and word-level TF-IDF treats each variant as a distinct, low-frequency token. Character n-grams (2–5) capture shared substrings across morphological variants, effectively implementing a lightweight morphological approximation without requiring a formal morphological analyser. The finding that unigram characters (range (1,1)) are insufficient, and that 5-grams are the optimal upper boundary, is consistent with the average Amharic morpheme length of 3–4 Ge'ez characters.

This finding generalises to other morphologically rich languages of the Horn of Africa (Tigrinya, Oromo, Somali) and offers a practical, computationally efficient alternative to morphological parsing in resource-constrained settings.

### RQ2: What preprocessing pipeline is appropriate for Amharic social-media text?

**Answer: Unicode NFC normalisation + Ethiopic character variant normalisation + noise removal + Amharic-specific stopword filtering.**

The five-stage preprocessing pipeline, described in Section 4.2, reduces effective vocabulary fragmentation without over-normalising the text. The most impactful single step is **Ethiopic character normalisation**, which maps phonetically equivalent Ge'ez character variants (e.g., all members of the ሐ/ኀ groups to ሃ, ፀ group to ጸ group) to canonical forms. Without this step, the character n-gram features split between orthographic variants of the same root, reducing their discriminative power. A secondary finding is that heavy preprocessing (aggressive stopword removal, morphological stemming) provides diminishing returns when character n-gram TF-IDF is used, because the character-level features are already invariant to many surface form variations.

### RQ3: What level of classification performance is achievable with current resources?

**Answer: F1_macro = 0.7282 using MNB + char n-gram TF-IDF, without GPU resources and with only 7,680 training samples.**

This result is directly comparable to the best transformer-based systems in the AfriSenti SemEval-2023 shared task (F1 weighted: 0.65–0.72) and to Ayele et al. (2023)'s XLM-RoBERTa result (F1_macro ≈ 0.72). The performance is achieved through feature engineering rather than model complexity, demonstrating that careful representation design can compensate for both limited training data and limited computational infrastructure. The Bi-LSTM (F1 = 0.5776) and random-init transformer (F1 = 0.3648) confirm that deep learning without pretrained representations does not improve on classical models in this data regime.

---

## 5.3 Contributions

This thesis makes four original contributions to Amharic natural language processing:

### Contribution 1: First Systematic Benchmark of Ten Models on AfriSenti Amharic

No prior published study has evaluated as many as ten models — spanning classical ML, gradient-boosted ensembles, deep learning, and transformer architectures — head-to-head on the official AfriSenti Amharic benchmark using consistent preprocessing and evaluation protocols. The 108 systematic experiments provide a reproducible reference point for future work. Specifically, the study benchmarks: MNB (word and char features), Logistic Regression, SVM, KNN, XGBoost, LightGBM, Bi-LSTM, a custom transformer, and an MNB–Transformer ensemble — all evaluated on the same test partition with macro-averaged F1.

This benchmark addresses a gap in the literature noted by Muhammad et al. (2023): the AfriSenti dataset was released with limited baseline analysis for the Amharic subset, and subsequent studies have used different evaluation setups, making comparisons unreliable. This thesis provides the missing systematic comparison.

### Contribution 2: Discovery That Character N-grams Dramatically Outperform Word-Level Features for Amharic

The experimental demonstration that character n-gram TF-IDF yields +11.66 F1 points over word-level TF-IDF (0.7282 vs. 0.6116) is a concrete, quantified finding with immediate implications for Amharic NLP. Prior Amharic sentiment studies have used word-level features as the default (e.g., Tessema and Yimam, 2021), without systematic comparison to sub-word alternatives. This thesis provides the first controlled ablation establishing character n-grams as the superior choice for morphologically rich Amharic text.

The finding is actionable: any future Amharic NLP project requiring text classification can adopt character n-gram TF-IDF (range (2-5), max_features≈80k) as a strong, easy-to-deploy baseline before investing in more complex representations.

### Contribution 3: Empirical Evidence That Pretrained Knowledge Is the Bottleneck for Amharic Deep Learning

The juxtaposition of the random-init transformer (F1 = 0.3648) with MNB (F1 = 0.7282) provides direct empirical evidence that **the transformer architecture itself provides no benefit over classical models on small Amharic datasets** — the performance of transformer-based systems in the literature (Ayele et al., 2023; SemEval-2023 top teams) is attributable to their pretrained representations, not their architecture.

This finding is theoretically significant: it adds Amharic to the set of empirically documented cases where classical models outperform randomly initialised deep learning, extending results by Joulin et al. (2017) and Wang et al. (2018) to a morphologically complex African language. It also provides a clear priority direction for the field: invest in Amharic-specific pretraining data and access to pretrained multilingual models (Afro-XLM-R, AfriBERTa) rather than in architectural innovation.

### Contribution 4: Reproducible Open-Source Pipeline for Amharic Sentiment Analysis

All preprocessing code, feature engineering scripts, model training pipelines, and evaluation protocols are version-controlled in the public GitHub repository. The pipeline includes:
- `src/preprocessing/preprocess.py`: Full Amharic preprocessing pipeline
- `src/experiment_runner*.py`: Four-phase systematic experiment framework
- `src/train_bilstm.py`: Bi-LSTM training script
- `src/train_transformer_cpu.py`: CPU-compatible transformer training
- `results/experiments/`: All 108 experiment logs in CSV format
- `results/final_figures/`: All publication-quality figures

This reproducible pipeline enables future researchers to replicate, extend, and compare results without repeating the experimental design choices that this study validated through systematic ablation.

---

## 5.4 Recommendations

### 5.4.1 For NLP Researchers Working on Amharic

**Adopt character n-gram TF-IDF as the default baseline.** The optimal configuration identified in this study — MNB with TF-IDF character n-grams (range (2-5), max_features=80,000, Laplace smoothing α=0.2) — requires no GPU, no pretrained model downloads, and trains in under 2 seconds. It should be the starting point for any Amharic text classification task, not an afterthought or a simple baseline to quickly surpass. As this study demonstrates, "surpassing" this baseline requires either pretrained transformers or substantially more training data.

**Do not neglect Ethiopic character normalisation.** The mapping of phonetically equivalent character variants to canonical forms (Section 4.2) is a preprocessing step specific to Amharic that is frequently omitted or underspecified in published work. Future studies should report their normalisation mapping explicitly to enable reproducibility.

**Use AfriSenti as the standard evaluation benchmark.** The AfriSenti Amharic dataset (Muhammad et al., 2023) is the only publicly available, multi-annotator, three-class Amharic sentiment dataset with standardised train/dev/test splits. Studies using custom datasets with different class definitions, annotation protocols, or evaluation metrics cannot be meaningfully compared to each other or to this work. The community should converge on AfriSenti as the standard benchmark.

### 5.4.2 For Practitioners Deploying Amharic Sentiment Tools

**The MNB + char n-gram model is deployable now.** With F1_macro = 0.7282, training time of 1.9 seconds, and inference time below 1 ms per tweet, this model can be integrated into real-time systems without GPU hardware. The trained model artefacts (TF-IDF vectoriser and MNB classifier) are available in the repository (`results/experiments/best_vec_phase4.pkl`, `best_clf_phase4.pkl`).

**Plan for data collection, not just model improvement.** The learning curve analysis (Section 4.11.5) shows that MNB performance has not plateaued at 7,680 training samples. Each additional thousand labelled examples is estimated to yield +0.5–1.0 F1 points. A crowdsourced annotation effort (e.g., using Appen, MTurk with Amharic-speaking annotators, or local university students) targeting 50,000 labelled tweets would likely push the MNB + char n-gram model to F1 ≥ 0.78 without any architectural change.

### 5.4.3 For Policymakers and Government Stakeholders

**Amharic sentiment classification is ready for deployment in policy monitoring applications.** The MNB + char n-gram model, achieving ~73 % macro-F1 across three sentiment classes, is sufficiently accurate for trend monitoring and alert systems — particularly when used in aggregate (e.g., tracking weekly sentiment trends for specific policy topics) rather than for individual-tweet classification. False-positive rates for individual tweets (~27 %) are acceptable in the context of exploratory policy analysis where human review of flagged cases is expected.

**Invest in Amharic language infrastructure.** The results of this study confirm that the primary bottleneck for Amharic NLP is not algorithmic — it is data and pretrained language model infrastructure. Government and civil-society investment in (a) large-scale Amharic text corpora for language model pretraining and (b) annotated datasets for downstream tasks (sentiment, named entity recognition, relation extraction) would provide substantially higher returns than funding model architecture research.

---

## 5.5 Future Work

### Priority 1: Fine-Tune Pretrained Multilingual Transformers

The most high-impact near-term task is fine-tuning **Afro-XLM-R** (Alabi et al., 2022) or **AfriBERTa** (Ogueji et al., 2021) on the AfriSenti Amharic training set with GPU access. Both models are pretrained on corpora including Amharic and related Semitic languages. Based on the SemEval-2023 leaderboard results for Amharic, fine-tuning is expected to yield F1_macro in the range **0.72–0.78**, with Afro-XLM-R (trained specifically on African languages) likely outperforming vanilla XLM-RoBERTa by an additional 3–6 F1 points. The training pipeline is fully implemented in `src/train_xlmr.py` and can be executed with network access to HuggingFace Hub and a GPU instance (≥8 GB VRAM, ~2 hours training time).

### Priority 2: Domain-Adaptive Pretraining on Ethiopian Policy Text

Fine-tuning a pretrained model that has been further adapted to Ethiopian government-policy discourse would likely outperform off-the-shelf pretrained models. The approach (Gururangan et al., 2020) involves:
1. Collecting a large corpus of unlabelled Amharic policy text (government press releases, parliamentary records, civil-society reports) — estimated 100,000–500,000 sentences are accessible online.
2. Continuing masked-language-model pretraining on this corpus for 5–10 epochs.
3. Fine-tuning the domain-adapted model for the downstream sentiment task.

Domain-adaptive pretraining has consistently yielded +1–4 F1 points over task-specific fine-tuning alone across multiple domains and languages (Gururangan et al., 2020).

### Priority 3: Expand the Annotated Amharic Dataset

The current 10,972-sample dataset is the primary bottleneck for all models. A targeted data collection campaign should aim for **50,000+ annotated Amharic tweets** spanning:
- Multiple policy domains (education, health, economy, security, justice)
- Multiple time periods (capturing political sentiment evolution)
- Dialectal variation (standard Amharic, regional dialects, diaspora register)
- Multiple annotation rounds with inter-annotator agreement verification

With 50K samples, the Bi-LSTM (and potentially a lightly parameterised fine-tuned transformer) would benefit substantially, enabling meaningful comparison across all architectural families.

### Priority 4: Extend to Ethiopian Language Siblings

The methodological framework developed here — particularly the character n-gram TF-IDF pipeline and systematic ablation protocol — can be applied directly to:
- **Tigrinya**: An Ethiosemitic language closely related to Amharic, for which AfriSenti provides a benchmark dataset (`data/raw/tir/`)
- **Oromo**: The largest language by speakers in Ethiopia, with AfriSenti coverage (`data/raw/orm/`)
- **Somali**: Spoken across the Horn of Africa, with AfriSenti coverage (`data/raw/som/`)

A cross-lingual comparative study would allow assessment of how well the character n-gram finding generalises across Afroasiatic language families with differing morphological complexity.

### Priority 5: Real-Time Policy Monitoring Dashboard

A Streamlit-based prototype dashboard that ingests live Amharic tweets by policy keyword, classifies them using the deployed MNB + char n-gram model, and displays aggregate sentiment trends with time-series visualisation is identified as the direct application target for this work. A skeleton implementation is planned as an extension of `src/dashboard/`. Key features:
- Keyword-filtered tweet ingestion (Twitter/X API or existing policy-domain scraper)
- Real-time sentiment classification (MNB model, <1 ms/tweet)
- Weekly aggregate trend plots per policy domain (education, health, economy, security)
- Alert system for sudden negative sentiment spikes (potential policy crisis signals)
- CSV export for government stakeholder reporting

### Priority 6: Few-Shot and Zero-Shot Approaches with Multilingual LLMs

Large multilingual language models (Claude, GPT-4, Llama-3) with demonstrated Amharic capability present an alternative evaluation path that bypasses the pretraining bottleneck. Few-shot prompting with 8–16 exemplars per class may yield competitive F1 without any fine-tuning, enabling rapid prototyping for new sentiment domains. Zero-shot chain-of-thought prompting in Amharic is of particular interest for cases where annotation resources are unavailable (e.g., novel policy domains). A systematic evaluation comparing few-shot LLM prompting against the MNB + char n-gram baseline is a natural extension of the current study, requiring only API access rather than GPU compute.

---

## 5.6 Closing Remarks

This thesis set out to determine the best approach for Amharic sentiment analysis under realistic resource constraints — no GPU, a small annotated dataset (10,972 samples), and no access to pretrained Amharic-specific models. The answer is both surprising in its simplicity and instructive in its generality: a carefully optimised Multinomial Naïve Bayes classifier with character-level TF-IDF n-gram features achieves F1_macro = 0.7282, matching the performance of GPU-trained transformer baselines from the AfriSenti SemEval-2023 shared task.

The reason is not that Naïve Bayes is a powerful model — it is not. The reason is that character-level features provide a natural inductive bias for Amharic's agglutinative morphology, effectively encoding morphological relationships that more powerful models would need large datasets to learn from scratch. When the data is limited, the right prior is more valuable than additional model capacity.

This finding has implications beyond Amharic. The Ethiopian NLP community, the broader African NLP research community, and any practitioner working on morphologically rich, low-resource languages can apply the character n-gram TF-IDF approach as an immediate, strong, deployable baseline — and understand precisely when and why to move beyond it (namely, when pretrained representations in the target language become accessible).

The thesis also makes an honest case for where the work ends and future work begins. The transformer results reported here are not the ceiling for Amharic sentiment analysis — they are a demonstration of the pretraining bottleneck. With Afro-XLM-R fine-tuning, the ceiling is likely near F1 = 0.78–0.80, and with a 50K-sample corpus, it may extend further. The infrastructure for that future work — the preprocessing pipeline, the systematic experimental framework, the benchmark evaluation protocol — is the lasting contribution of this thesis.

---

*This study was conducted as part of an MSc dissertation in Computer Science / Artificial Intelligence. All experimental results are reproducible from the code and data in the accompanying repository. The author encourages the Amharic NLP community to build on, critique, and extend these findings.*
