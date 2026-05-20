# Responses to Advisor Comments — Dr. Martha Yifiru

**Thesis:** Developing Deep Learning-Based Sentiment Analysis of Amharic Social Media for Public Policy Enhancement in Ethiopia  
**Author:** Getnet Bogale, Addis Ababa University  
**Date:** 2026-05-20

---

## Comment 1 — "How are you going to answer this question?"

**Attached to Research Question:** *Which sentiment classification model (ML, deep learning, or transformer-based) achieves the best macro-averaged F1 score on Amharic social-media text in a public-policy domain?*

**Response:**

This question is answered empirically through a controlled comparative experiment. Six classifiers — Naïve Bayes, Logistic Regression, SVM, KNN, Bi-LSTM, and XLM-RoBERTa — are trained on an identical pre-processed dataset (7,680 training samples from AfriSenti + supplementary policy tweets) and evaluated on a common held-out test partition (n = 1,646, stratified). Each model is assessed using macro-averaged F1, which weights all three sentiment classes equally regardless of support, making it the appropriate metric for an imbalanced corpus. The model that achieves the highest macro-F1 on the test set is identified as the best-performing architecture for this language and domain. This direct experimental comparison provides a clear, replicable answer to the question.

---

## Comment 2 — "Can't you get the answer for this from literature?"

**Attached to Hypothesis/Question:** *Does preprocessing (Unicode normalisation, noise removal, stopword filtering) improve classifier performance on Amharic social-media text compared to raw input?*

**Response:**

The general benefit of preprocessing for NLP is well-established in the literature. The specific Ethiopic character normalisation step — collapsing the three visually equivalent groups (ሀ/ሐ/ኀ, ዐ/አ, ፀ/ጸ) to canonical forms — is documented in prior Amharic NLP work (Yimam et al., 2018; Ayele et al., 2023) and its utility can therefore be partially justified from existing sources. However, the *quantitative impact* of this specific preprocessing combination on *policy-domain Amharic Twitter data* cannot be read from literature alone, because: (a) no prior study has applied this exact pipeline to the AfriSenti Amharic benchmark combined with supplementary policy tweets; (b) the magnitude of vocabulary reduction (and its effect on TF-IDF feature quality) depends on the specific corpus; and (c) the relative benefit of each preprocessing step may differ for character-level models (Bi-LSTM) vs. word-level models (Naïve Bayes) vs. sub-word models (XLM-RoBERTa). The empirical component is therefore necessary to quantify the effect in this specific experimental setting, even where the directional expectation is supported by prior literature.

---

## Comment 3 — "What will be the difference between your work and the previous works?"

**Response:**

This thesis makes three concrete contributions that are not present in any single prior study on Amharic sentiment analysis:

**Contribution 1: First systematic benchmark of six model families on the same standardised Amharic test set.**  
Existing work evaluates subsets of model families in isolation. Ayele et al. (2023) compare transformer models; Tessema and Yimam (2021) focus on traditional ML; Alemayehu et al. (2023) examine a single-domain political corpus. No published study, to the best of the researcher's knowledge, has placed Naïve Bayes, Logistic Regression, SVM, KNN, Bi-LSTM, and XLM-RoBERTa in a head-to-head comparison using the same standardised AfriSenti Amharic evaluation partition, the same preprocessing pipeline, and the same evaluation metric. This study provides that benchmark, making it possible for future researchers to accurately position new models relative to a common baseline.

**Contribution 2: First application of XLM-RoBERTa/multilingual transformers to *policy-domain* Amharic sentiment specifically.**  
Prior transformer-based Amharic sentiment studies have used general social-media data. This study explicitly targets public-policy discourse (education, health, economy, security), which involves domain-specific vocabulary and a different sentiment distribution from celebrity or sports commentary. The policy-domain focus is practically motivated: the downstream application is a government policy-monitoring dashboard, not a general-purpose tool.

**Contribution 3: A reproducible, end-to-end pipeline with a policy-decision-support output layer.**  
This study contributes not only a trained model but a complete, version-controlled pipeline from raw tweet collection through preprocessing, model training, and evaluation, to a conceptual dashboard framework that translates model predictions into trend visualisations for non-technical government stakeholders. The pipeline is fully documented (`src/`, `configs/model_config.yaml`, `scripts/run_full_pipeline.py`) and the methodology diagram (`results/figures/methodology_pipeline.pdf`) makes the research design reproducible. Prior Amharic sentiment studies do not include this applied decision-support component.

---

## Comment 4 — "How do you develop a classification model without annotated data? How is it possible to get a framework from the model development process? How will you evaluate a model without using the model? How do you get policy decision support from model evaluation?"

**Acknowledgement:**

The advisor's critique is valid. The original methodology section described steps that were not logically connected: it listed data collection, model training, evaluation, and policy decision support as separate activities without making the causal dependencies between them explicit. The result was a methodology that appeared to develop models without data, evaluate models without training them, and generate policy insights without deploying a working model. This has been corrected.

**Corrected Pipeline (see Figure X.X — `results/figures/methodology_pipeline.pdf`):**

The revised methodology is a strictly sequential pipeline in which each stage depends on the completion of the previous one:

1. **Raw Amharic social-media data** (Twitter/X, Facebook, Telegram) is *collected first*, before any modelling activity.

2. **Data collection produces a raw corpus** — AfriSenti (~8,950 pre-labelled tweets) plus ~2,530 additional policy-domain tweets collected using keyword search. The AfriSenti labels were produced by trained annotators and published with the shared task; the supplementary tweets were annotated using the same three-class scheme (positive, negative, neutral).

3. **Preprocessing** (Unicode normalisation, Ethiopic character normalisation, noise removal, tokenisation, stopword filtering) converts the raw annotated corpus into a clean, consistent feature representation. *Annotation is a prerequisite for preprocessing, not a product of it.*

4. **The annotated, preprocessed dataset is split** (70/15/15, stratified) into training, validation, and test partitions. Only the training partition is visible to the models during learning.

5. **Model training** is conducted on the training partition. Each of the six models (Naïve Bayes, Logistic Regression, SVM, KNN, Bi-LSTM, XLM-RoBERTa) learns from the labelled training samples. *A model cannot be trained without annotated data — the labels are the supervision signal.*

6. **Evaluation** is performed on the held-out test partition using the trained models. *The model must be fully trained before evaluation; evaluation does not precede or substitute for training.* Performance metrics (accuracy, precision, recall, macro-F1) are computed by comparing model predictions against the known test labels.

7. **Best model selection** identifies the architecture achieving the highest macro-F1 on the test set.

8. **Sentiment trend analysis** applies the selected model to new, unlabelled Amharic policy tweets to generate real-time sentiment predictions.

9. **Policy decision support** visualises these predictions as time-series sentiment trends for government stakeholders. *Policy insight is derived from model predictions on live data — it is not derived from evaluation metrics alone.* The dashboard enables monitoring of public sentiment about specific policy domains over time, providing actionable signals to decision-makers.

**Summary of dependencies:** Annotation → Preprocessing → Split → Training → Evaluation → Best Model → Deployment → Policy Insight. Every arrow in this chain represents a hard prerequisite; no stage can be bypassed or reordered. The revised methodology diagram (`results/figures/methodology_pipeline.pdf`) makes these dependencies visually explicit.
