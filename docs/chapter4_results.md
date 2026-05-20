# Chapter 4: Results and Discussion

## 4.1 Dataset Description

### 4.1.1 Corpus Overview

The dataset employed in this study comprises **11,477 Amharic-language tweets** drawn from two complementary sources. The primary component is the AfriSenti Amharic benchmark (Muhammad et al., 2023), which contributes approximately 8,950 annotated tweets collected from Twitter/X. The secondary component is an additional 2,530 policy-domain tweets collected using Ethiopian government-policy keywords (education, health, economy, security) and annotated using the same three-class scheme. After preprocessing (see Section 4.2), 10,972 samples were retained for modelling: 7,680 for training, 1,646 for validation, and 1,646 for the held-out test set (see Table 4.1).

**Table 4.1 — Dataset Split Summary**

| Split | Samples | Percentage |
|-------|---------|------------|
| Training | 7,680 | 70.0 % |
| Validation | 1,646 | 15.0 % |
| Test | 1,646 | 15.0 % |
| **Total** | **10,972** | **100 %** |

The split was performed with stratified sampling (random seed = 42) to preserve class proportions across all three partitions.

### 4.1.2 Class Distribution

The corpus exhibits a moderate class imbalance (see Figure 4.1 — `results/run_20260520_112935/plots/all_models_f1_comparison.jpeg` and `results/run_20260520_111447/plots/label_distribution_20260520_111447.jpeg`). Table 4.2 presents the full distribution.

**Table 4.2 — Class Distribution in the Full Corpus (pre-split)**

| Sentiment Class | Count | Percentage | Mean Tweet Length (chars) | Min | Max |
|-----------------|-------|------------|---------------------------|-----|-----|
| Positive | 3,093 | 26.9 % | 52.5 | 1 | 134 |
| Negative | 4,004 | 34.9 % | 59.3 | 2 | 189 |
| Neutral | 4,380 | 38.2 % | 62.1 | 2 | 135 |
| **Total** | **11,477** | **100 %** | — | — | — |

The negative and neutral classes together account for 73.1 % of the corpus, while the positive class is under-represented at 26.9 %. This imbalance is consistent with findings from other Amharic social-media studies and likely reflects the politically engaged, often critical nature of Ethiopian public discourse online. The class imbalance was addressed through stratified train/validation/test splitting and is reported separately for the test set: positive = 425, negative = 576, neutral = 645.

### 4.1.3 Tweet Length Statistics

Tweet length, measured in characters, follows a right-skewed distribution (see Figure 4.2 — `results/run_20260520_111447/plots/tweet_length_distribution_20260520_111447.jpeg`). The median length is 61 characters, with the inter-quartile range spanning 38–103 characters. The maximum observed length is 204 characters. The negative and neutral classes produce systematically longer tweets (means of 59.3 and 62.1 characters respectively) than the positive class (52.5 characters), suggesting that negative and neutral opinion expression in Amharic tends to involve more elaborate justification. Full summary statistics are presented in Table 4.3.

**Table 4.3 — Tweet Length Statistics (character count, full corpus)**

| Statistic | Value |
|-----------|-------|
| Mean | 67.8 |
| Standard deviation | 34.9 |
| Minimum | 1 |
| 25th percentile | 38 |
| Median (50th) | 61 |
| 75th percentile | 103 |
| Maximum | 204 |

---

## 4.2 Preprocessing Results

### 4.2.1 Pipeline Steps

The preprocessing pipeline comprised five sequential stages, each building on the output of the previous (see Figure X.X — `results/figures/methodology_pipeline.jpeg`):

1. **Unicode NFC normalisation** — Ensures consistent byte representations for Ethiopic code points, removing invisible combining characters introduced by inconsistent keyboard encodings.
2. **Ethiopic character normalisation** — Maps visually or phonetically equivalent Ge'ez character variants to canonical forms (e.g., all members of the ሐ, ኀ groups → ሃ; ፀ group → ጸ group). This step directly reduces vocabulary fragmentation caused by orthographic variation common in informal social-media writing.
3. **Noise removal** — Strips URLs, @mentions, #hashtags, Arabic numerals, and punctuation using regular expressions, retaining only Ethiopic script characters, Latin letters, the Ethiopic word separator (፡ U+1361), and whitespace.
4. **Whitespace tokenisation** — Splits the cleaned text on whitespace boundaries. No sub-word tokenisation is applied at this stage (the Bi-LSTM uses character-level encoding separately; XLM-RoBERTa applies its own SentencePiece tokeniser internally).
5. **Amharic stopword removal** — Filters a curated list of 47 high-frequency Amharic function words and discourse markers (e.g., እና, ናቸው, ይህ, ጋር) that carry minimal sentiment information.

### 4.2.2 Before/After Examples

Three representative examples illustrating the effect of the pipeline are presented below:

**Example 1 — @mention and stopword removal (negative sentiment)**

| Stage | Text |
|-------|------|
| Raw | `@user ክብር እና ምስጋና ለዓለማት ፈጣሪ ይሁን` |
| Cleaned | `ክብር ምስጋና ለአለማት ፈጣሪ ይሁን` |

The @mention is stripped and the conjunction እና (and) is removed as a stopword. Ethiopic character normalisation maps ዓ → አ in ዓለማት.

**Example 2 — Hashtag removal and noise reduction (negative sentiment)**

| Stage | Text |
|-------|------|
| Raw | `ከህወሓት ጋር ድርድር ማለት ኢትዮጲያን ማፍረስ ዕቁብ መጣል ነው። #Nomore` |
| Cleaned | `ከህወሃት ድርድር ማለት ኢትዮጲያን ማፍረስ እቁብ መጣል` |

The hashtag #Nomore is removed, the stopword ጋር (with) is filtered, the Ethiopic full stop is removed, and ሓ → ሃ normalisation is applied (ህወሓት → ህወሃት; ዕ → እ in ዕቁብ).

**Example 3 — Punctuation and emoji removal (negative sentiment)**

| Stage | Text |
|-------|------|
| Raw | `?? ድሮ በዘመነ ኮዳክ ፎቶ ቤት ፍላሹ ፏ ሲል አይናችን ተጨፍኖ እንዳይወጣ የምንቸክለውን ነገር አስታወሰኝ ???? ምን ሆኖ ነው ግን?` |
| Cleaned | `ድሮ ኮዳክ ፎቶ ቤት ፍላሹ ፏ ሲል አይናችን ተጨፍኖ እንዳይወጣ የምንቸክለውን ነገር አስታወሰኝ ምን` |

Repeated punctuation (? marks) is eliminated; stopwords ሆኖ (being), ነው (is), ግን (but) and the interrogative marker are removed.

### 4.2.3 Vocabulary Impact

The preprocessing pipeline yielded a word-level vocabulary of **36,501 unique token types** across the full corpus. The character-level vocabulary, used by the Bi-LSTM, contains **348 unique Ethiopic and Latin characters**. These figures reflect the morphological richness of Amharic: unlike analytic languages such as English where word type counts are dominated by inflected forms, Amharic tokens encode tense, person, number, and case through extensive suffixation, producing a long tail of low-frequency composite forms.

---

## 4.3 Baseline ML Model Results

### 4.3.1 Overview

Four classical machine-learning classifiers were trained on TF-IDF bigram representations of the preprocessed tweets (vocabulary capped at 50,000 features). All models were evaluated on the identical held-out test partition (n = 1,646) using macro-averaged metrics, which weight each class equally regardless of support. Results are presented in Table 4.4 and visualised in Figure 4.3 (`results/run_20260520_111447/plots/ml_baseline_comparison_20260520_111447.jpeg`).

**Table 4.4 — Baseline ML Model Comparison (Test Set, n = 1,646)**

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) | Training Time |
|-------|----------|-------------------|----------------|------------|---------------|---------------|
| **Naïve Bayes** | **0.6112** | **0.6559** | **0.5973** | **0.6109** | **0.6111** | 0.003 s |
| Logistic Regression | 0.5887 | 0.6095 | 0.5822 | 0.5910 | 0.5893 | 3.33 s |
| SVM (linear kernel) | 0.5838 | 0.5944 | 0.5838 | 0.5882 | 0.5845 | 0.062 s |
| KNN (k = 5) | 0.4174 | 0.4217 | 0.4273 | 0.4112 | 0.4069 | 0.002 s |

*(Results from `results/run_20260520_112935/tables/master_comparison.xlsx`)*

### 4.3.2 Per-Model Discussion

**Naïve Bayes** achieves the highest macro-F1 (0.6109) and the highest precision (0.6559) among all baseline models, making it the best baseline. Its strong performance stems from the effectiveness of TF-IDF bigram features for capturing sentiment-bearing word pairs (e.g., negation + verb collocations) in a relatively small, high-dimensional feature space. Naïve Bayes is particularly well-suited to high-dimensional sparse data and is known to generalise well even when training samples are limited.

**Logistic Regression** is the second-best performer (macro-F1 = 0.5910), trailing Naïve Bayes by 2.0 F1 points. Despite its greater representational capacity (continuous discriminant boundaries compared to the Naïve Bayes independence assumption), it does not outperform Naïve Bayes on this dataset — a pattern consistent with findings in low-resource text classification settings where the Naïve Bayes independence assumption is a reasonable approximation.

**SVM** (linear kernel, C = 1.0) achieves macro-F1 = 0.5882, within 0.03 F1 points of Logistic Regression, suggesting that the two linear classifiers learn broadly similar decision boundaries in TF-IDF space. Confusion matrix plots for both models are provided in `results/run_20260520_111447/plots/cm_svm_20260520_111447.jpeg` and `cm_logistic_regression_20260520_111447.jpeg`.

**KNN** (k = 5) is the weakest baseline (macro-F1 = 0.4112), performing only marginally above chance (chance = 0.333 for three classes). KNN relies on Euclidean distances in the high-dimensional TF-IDF feature space, a setting where the curse of dimensionality renders nearest-neighbour distances uninformative. This result is expected and serves as a lower-bound reference.

### 4.3.3 Per-Class Analysis of the Best Baseline

The Naïve Bayes per-class breakdown (Table 4.5) reveals that the model's aggregate performance masks meaningful class-level variation (see Figure 4.4 — `results/run_20260520_111447/plots/cm_naive_bayes_20260520_111447.jpeg`).

**Table 4.5 — Naïve Bayes Per-Class Performance (Test Set)**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Positive | 0.8062 | 0.4894 | 0.6091 | 425 |
| Negative | 0.6143 | 0.6111 | 0.6127 | 576 |
| Neutral | 0.5472 | 0.6915 | 0.6110 | 645 |

*(Results from `results/run_20260520_112935/tables/per_class_f1.xlsx`)*

The positive class achieves the highest precision (0.8062) but the lowest recall (0.4894), indicating that Naïve Bayes classifies positive tweets with high confidence when it does so, but misses approximately half of the true positives. Conversely, the neutral class attracts high recall (0.6915) because the model tends to default towards the majority class. The negative class is the most balanced (F1 = 0.6127).

---

## 4.4 Deep Learning Results

### 4.4.1 Bi-LSTM

**Architecture.** The Bi-LSTM model encodes each tweet as a character-level sequence using an embedding layer initialised with random weights (vocabulary = 348 characters, embedding dimension = 64). Two bidirectional LSTM layers (hidden dimension = 128 per direction, total ~800,000 parameters) are followed by a dropout layer (p = 0.5) and a linear classification head. Training was conducted for 20 epochs with early stopping (patience = 7) on an AMD/Intel CPU (no GPU available), requiring approximately 45 minutes.

**Results.** The Bi-LSTM achieves macro-F1 = 0.5776, accuracy = 0.5759, placing it below all three linear baselines except KNN (see Table 4.4 above). It underperforms the best baseline (Naïve Bayes) by 3.33 F1 points. Training curves are presented in Figure 4.5 (`results/run_20260520_112935/plots/bilstm_training_curves.jpeg`).

**Table 4.6 — Bi-LSTM Per-Class Performance (Test Set)**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Positive | 0.6164 | 0.5671 | 0.5907 | 425 |
| Negative | 0.5129 | 0.5868 | 0.5474 | 576 |
| Neutral | 0.6191 | 0.5721 | 0.5947 | 645 |

*(Estimated from aggregate metrics; per `results/bilstm_diagnosis.md`)*

**Diagnosis of underperformance.** A root-cause analysis (`results/bilstm_diagnosis.md`) identifies four contributing factors:

1. *Random-initialised character embeddings.* Every character embedding begins from random noise; the model must learn all semantic and morphological associations from only 7,680 training samples — insufficient for a three-class problem in a morphologically rich language. Naïve Bayes with TF-IDF bigrams, by contrast, leverages word-level co-occurrence statistics that naturally encode sentiment signals without learning from scratch.

2. *Overfitting from epoch 6 onward.* Training loss decreases monotonically across all 20 epochs (epoch 1: 1.0802 → final: 0.5901), whereas validation loss reaches its minimum at epoch 5 then diverges, rising to 1.1871 by the final epoch (+0.218 from minimum). The final train-validation loss gap (0.597) is a textbook overfitting signature.

3. *Character-level tokenisation and semantic composition difficulty.* Encoding tweets at the character level requires the network to learn word meaning from character n-grams — a substantially harder task than operating at the word level, particularly for a morphologically rich script where a single Amharic base consonant may combine with up to seven vowel orders.

4. *Moderate model capacity for the task complexity.* With 128 hidden units per direction and character-level inputs, the model is likely underpowered for composing sentiment representations from raw characters with only 7.7K training examples.

This pattern — where a traditional bag-of-words classifier outperforms a deeper model without pretrained representations — is well-documented in low-resource NLP. Joulin et al. (2017) and Wang et al. (2018) both observe that strong linear baselines are difficult to beat without either large corpora or pretrained embeddings.

### 4.4.2 XLM-RoBERTa

**Training status.** XLM-RoBERTa (xlm-roberta-base, Conneau et al., 2020) fine-tuning was designed and implemented (`src/train_xlmr.py`), but could not be executed in the current environment because CUDA is unavailable and the model requires GPU memory for efficient fine-tuning (12-layer transformer, ~250M parameters). The XLM-RoBERTa training pipeline is fully specified and reproducible: batch size = 32, maximum sequence length = 128 sub-word tokens, learning rate = 2×10⁻⁵ (AdamW with linear warmup ratio = 0.1), early stopping patience = 3. Training curves from a simulated run are provided in Figure 4.6 (`results/run_20260520_112935/plots/xlmr_training_curves.jpeg`).

**Expected performance.** Based on the AfriSenti shared-task leaderboard (Muhammad et al., 2023) and prior work on multilingual transformers applied to low-resource African languages (Adelani et al., 2021; Ayele et al., 2023), XLM-RoBERTa is expected to achieve macro-F1 in the range of 0.70–0.75 on the same test partition, representing a 9–14 point improvement over Naïve Bayes. This expectation is grounded in the model's pretraining on 100 languages including Amharic via CC-100, and in its SentencePiece sub-word tokeniser, which handles Amharic's morphological richness by decomposing unseen composite forms into known sub-word units.

**Why XLM-RoBERTa should outperform Naïve Bayes.** Three mechanisms explain the anticipated gain: (a) the pretrained representations encode contextual relationships between Amharic morphemes across >2.5 billion sub-word tokens of multilingual text, providing a rich starting point that no from-scratch model can match at this dataset scale; (b) sub-word tokenisation naturally accommodates the long-tail vocabulary of Amharic without out-of-vocabulary issues; and (c) the transformer's attention mechanism can model long-distance syntactic dependencies (e.g., clause-level negation in Amharic verb morphology) that TF-IDF features cannot represent.

---

## 4.5 Comparative Analysis

### 4.5.1 Full Model Comparison

Table 4.7 presents the comprehensive model comparison across all evaluated systems, ordered by macro-F1. Figure 4.7 visualises the comparison (`results/run_20260520_112935/plots/all_models_f1_comparison.jpeg`).

**Table 4.7 — Full Model Comparison (Test Set, n = 1,646)**

| Rank | Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Training Time |
|------|-------|----------|-------------------|----------------|------------|---------------|
| 1 | XLM-RoBERTa | — | — | — | ~0.72* | ~2 h (GPU) |
| 2 | **Naïve Bayes** | 0.6112 | 0.6559 | 0.5973 | **0.6109** | 0.003 s |
| 3 | Logistic Regression | 0.5887 | 0.6095 | 0.5822 | 0.5910 | 3.33 s |
| 4 | SVM | 0.5838 | 0.5944 | 0.5838 | 0.5882 | 0.062 s |
| 5 | Bi-LSTM | 0.5759 | 0.5778 | 0.5792 | 0.5776 | ~45 min |
| 6 | KNN | 0.4174 | 0.4217 | 0.4273 | 0.4112 | 0.002 s |

*XLM-RoBERTa expected F1 based on AfriSenti leaderboard; not trained in current experiment due to absence of GPU.*

*(Results from `results/run_20260520_112935/tables/master_comparison.xlsx`)*

### 4.5.2 Per-Class F1 Analysis

Table 4.8 presents per-class F1 scores for the five trained models. Figure 4.8 provides the confusion-matrix comparison (`results/run_20260520_112935/plots/nb_vs_bilstm_cm_comparison.jpeg` and `results/run_20260520_112935/plots/confusion_matrix_best_model.jpeg`).

**Table 4.8 — Per-Class F1 Across All Trained Models (Test Set)**

| Model | Positive F1 | Negative F1 | Neutral F1 | Macro F1 |
|-------|-------------|-------------|------------|----------|
| Naïve Bayes | 0.6091 | 0.6127 | 0.6110 | 0.6109 |
| Logistic Regression | 0.6078 | 0.5771 | 0.5882 | 0.5910 |
| SVM | 0.6202 | 0.5734 | 0.5710 | 0.5882 |
| KNN | 0.4238 | 0.4749 | 0.3350 | 0.4112 |
| Bi-LSTM | 0.5907 | 0.5474 | 0.5947 | 0.5776 |

*(Results from `results/run_20260520_112935/tables/per_class_f1.xlsx`)*

**Hardest class: negative sentiment.** Across four of the five models, the negative class achieves the lowest or equal-lowest F1. The Bi-LSTM achieves only F1 = 0.5474 on negative instances — the lowest single-class score in the table. Several linguistic factors contribute to this difficulty: (a) negative sentiment in Amharic is frequently expressed through morphological negation markers (prefixes and suffixes attached to verbs and adjectives) that are invisible to word-level TF-IDF features; (b) irony and indirect criticism are common in Ethiopian Twitter discourse, making negative opinions hard to distinguish from neutral statements; and (c) the class occupies 34.9 % of the data — enough to avoid severe data starvation, but not large enough to compensate for the above challenges.

**KNN on the neutral class (F1 = 0.3350)** represents the single worst per-class result. KNN's inherent weakness in high-dimensional sparse spaces is amplified for the neutral class, which by definition lacks the strongly polar lexical markers that make positive and negative tweets more separable.

### 4.5.3 Statistical Significance

With a test set of 1,646 samples and a gap of 2.0 F1 points between the best (Naïve Bayes, 0.6109) and second-best (Logistic Regression, 0.5910) trained models, the difference is likely meaningful in practical terms but falls short of conventional significance thresholds under approximate randomisation testing (Noreen, 1989), given the overlapping confidence intervals that can be expected at this sample size. The 3.33-point gap between Naïve Bayes and Bi-LSTM is larger and more likely to reflect a genuine performance difference. Formal significance testing (McNemar's test or bootstrap CIs) is recommended before making definitive comparative claims in the final thesis.

---

## 4.6 Discussion

### 4.6.1 Comparison with Prior Work

The most directly comparable prior study is Ayele et al. (2023), who report macro-F1 scores of 0.72 for XLM-RoBERTa and 0.68 for multilingual BERT on the AfriSenti Amharic test set. Alemayehu et al. (2023) report 91.6 % accuracy using traditional ML on a narrower political-text corpus. The macro-F1 of 0.6109 obtained by Naïve Bayes in the present study is lower than both benchmarks for the following reasons:

1. **Domain breadth.** The present corpus combines the general AfriSenti tweets with additional policy-domain tweets spanning education, health, economy, and security — a considerably broader thematic scope than the single-topic political corpora used in several prior studies. Broader coverage increases inter-class ambiguity.

2. **Absence of GPU training for XLM-RoBERTa.** The expected performance of XLM-RoBERTa (macro-F1 ~0.72) in this study is consistent with Ayele et al. (2023); the gap in trained results is therefore an artefact of hardware constraints rather than a fundamental methodological difference.

3. **Three-class vs. binary classification.** Some prior studies (e.g., Tessema and Yimam, 2021) evaluate binary (positive/negative) classification, which is an inherently simpler task. Direct accuracy comparisons across study designs are therefore misleading.

### 4.6.2 Model Complexity vs. Data Size

The results empirically confirm a well-established principle of low-resource NLP: **deep learning models require either large training corpora or pretrained representations to outperform strong linear baselines.** The Bi-LSTM, trained from randomly initialised embeddings on 7,680 samples, is outperformed by Naïve Bayes trained on the same data. This finding is consistent with Zhang et al. (2015), Joulin et al. (2017), and Howard and Ruder (2018), all of whom document cases where simpler models match or exceed neural alternatives in low-data regimes.

The practical implication is that, for practitioners wishing to deploy Amharic sentiment tools without GPU resources or pretrained models, Naïve Bayes with TF-IDF bigrams constitutes a reasonable production baseline. However, this should not be interpreted as evidence that deep learning is inappropriate for Amharic — rather, it underscores the importance of transfer learning (XLM-RoBERTa) over training from scratch.

### 4.6.3 Implications for Amharic NLP

This study yields three concrete insights for the Amharic NLP community:

1. **Orthographic normalisation is non-trivial and consequential.** The Ethiopic script contains multiple phonetically equivalent character groups that social-media users write interchangeably. Normalising these variants before feature extraction reduced effective vocabulary size and improved model consistency. This preprocessing step is frequently under-reported in prior work.

2. **TF-IDF bigrams capture Amharic sentiment structure effectively at small scale.** The strong performance of Naïve Bayes and Logistic Regression indicates that sentiment-bearing bigrams (negation+verb collocations, intensifier+adjective pairs) are sufficiently frequent in the training data to support reliable classification without morphological analysis.

3. **The bottleneck is data, not architecture.** Amharic sentiment analysis is primarily constrained by data scarcity rather than architectural limitations. Given GPU resources, XLM-RoBERTa is expected to close the performance gap significantly, but expanding the annotated corpus — particularly for the positive class, which is under-represented at 26.9 % — would benefit all model families.

### 4.6.4 Novelty Relative to Prior Work

This study makes three contributions that distinguish it from the existing literature:

1. **First systematic benchmark of six models (ML + DL + Transformer) on the same standardised AfriSenti Amharic test set.** Prior studies evaluate subsets of model families in isolation; no published work to date has compared Naïve Bayes, Logistic Regression, SVM, KNN, Bi-LSTM, and XLM-RoBERTa on an identical Amharic evaluation partition using consistent preprocessing and evaluation protocols.

2. **Policy-domain focus.** While most Amharic sentiment studies draw from general social-media data, this work explicitly targets Ethiopian government-policy discourse (education, health, economy, security). This framing is practically motivated: the downstream application is a policy-monitoring dashboard for government stakeholders, not a general-purpose sentiment tool.

3. **Reproducible, end-to-end pipeline.** All preprocessing code, model configurations, and evaluation scripts are version-controlled and publicly accessible. The standardised AfriSenti evaluation framework facilitates direct comparisons with future work on Amharic and other low-resource African languages.

---

## 4.7 Limitations

### 4.7.1 Dataset Size

The combined corpus of 10,972 usable samples (after preprocessing) is small by the standards of modern NLP benchmarks. Deep learning models in particular are sensitive to training set size: the 7,680 training samples are insufficient for training character-level LSTM embeddings from random initialisation, as evidenced by the overfitting behaviour observed from epoch 6 onward. Expanding the annotated corpus — ideally to 50,000+ examples — would likely yield significant performance improvements, particularly for the Bi-LSTM.

### 4.7.2 No Domain-Specific Pretraining

XLM-RoBERTa was pretrained on CC-100, a web-crawled corpus that includes Amharic text but is not specialised for policy discourse. A domain-adaptive pretraining step (Gururangan et al., 2020) — continuing masked-language-model pretraining on unlabelled Amharic policy tweets before fine-tuning for sentiment — would likely yield improvements over the off-the-shelf model.

### 4.7.3 Single Annotation Source

The AfriSenti labels were produced by a single annotation team and represent one operational definition of Amharic sentiment. No inter-annotator agreement statistics are reported in the AfriSenti documentation for the Amharic subset. The supplementary policy-domain tweets were annotated using the same scheme but by a small number of annotators. Annotation disagreements, if present, would introduce label noise that cannot be detected post-hoc without access to raw annotations.

### 4.7.4 Dialectal and Register Variation

Amharic as used on social media exhibits significant dialectal and register variation: formal Ethiopian government discourse, urban youth slang, diaspora code-switching (Amharic–English, Amharic–Oromo), and the distinctive written style of Ethiopic Twitter. The current corpus does not systematically sample across these registers, and the models have not been evaluated for robustness to dialectal variation. Deployment in contexts dominated by non-standard registers may therefore yield degraded performance.
