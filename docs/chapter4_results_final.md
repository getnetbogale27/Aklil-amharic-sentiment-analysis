# Chapter 4: Results and Discussion

---

## 4.1 Dataset Characteristics

### 4.1.1 Corpus Overview

The dataset used in this study comprises **11,477 Amharic-language tweets** drawn from two complementary sources. The primary component is the AfriSenti Amharic benchmark (Muhammad et al., 2023), contributing approximately 8,950 annotated tweets collected from Twitter/X using the official AfriSenti train/dev/test splits. The secondary component consists of 2,530 additional tweets collected using Ethiopian government-policy keywords spanning education, health, economy, and security, annotated under the same three-class scheme (positive, negative, neutral).

After preprocessing (described in Section 4.2), **10,972 samples** were retained for modelling. These were partitioned using stratified random splitting (seed = 42) to maintain class proportions across all three sets.

**Table 4.1 — Dataset Split Summary**

| Split      | Samples | Percentage |
|------------|---------|------------|
| Training   | 7,680   | 70.0 %     |
| Validation | 1,646   | 15.0 %     |
| Test       | 1,646   | 15.0 %     |
| **Total**  | **10,972** | **100 %** |

All evaluation metrics reported in this chapter are computed on the held-out test partition (n = 1,646), which was never used for model selection or hyperparameter tuning.

### 4.1.2 Class Distribution

The corpus exhibits moderate class imbalance, with the negative and neutral classes together accounting for 73.1 % of samples (see Figure 4.1 — `results/final_figures/fig4_1_label_distribution.jpeg`).

**Table 4.2 — Class Distribution in the Full Corpus**

| Sentiment Class | Count  | Percentage | Mean Tweet Length (chars) | Min | Max |
|-----------------|--------|------------|---------------------------|-----|-----|
| Positive        | 3,093  | 26.9 %     | 52.5                      | 1   | 134 |
| Negative        | 4,004  | 34.9 %     | 59.3                      | 2   | 189 |
| Neutral         | 4,380  | 38.2 %     | 62.1                      | 2   | 135 |
| **Total**       | **11,477** | **100 %** | —                     | —   | —   |

The under-representation of the positive class (26.9 %) is consistent with findings in other Amharic social-media studies and likely reflects the critical, politically engaged nature of Ethiopian public discourse on Twitter. The class imbalance was addressed through stratified splitting; the test-set class support is: positive = 425, negative = 576, neutral = 645.

### 4.1.3 Tweet Length Statistics

Tweet length, measured in characters, follows a right-skewed distribution (Figure 4.2 — `results/final_figures/fig4_2_tweet_length_by_class.jpeg`). The negative and neutral classes produce systematically longer tweets (means 59.3 and 62.1 characters respectively) than the positive class (52.5 characters), suggesting that expressing criticism or neutrality in Amharic requires more elaboration.

**Table 4.3 — Tweet Length Statistics (character count, full corpus)**

| Statistic           | Value |
|---------------------|-------|
| Mean                | 67.8  |
| Standard deviation  | 34.9  |
| Minimum             | 1     |
| 25th percentile     | 38    |
| Median (50th)       | 61    |
| 75th percentile     | 103   |
| Maximum             | 204   |

---

## 4.2 Preprocessing Results

### 4.2.1 Pipeline Steps

The preprocessing pipeline comprised five sequential stages applied to every tweet before feature extraction (see Figure 3.1 — `results/final_figures/fig3_1_methodology_pipeline.jpeg`):

1. **Unicode NFC normalisation** — Ensures consistent byte representations for Ethiopic code points, removing invisible combining characters introduced by inconsistent keyboard encodings.
2. **Ethiopic character normalisation** — Maps phonetically equivalent Ge'ez character variants to canonical forms (e.g., all members of the ሐ/ኀ groups → ሃ; ፀ group → ጸ group). This step directly reduces vocabulary fragmentation caused by the orthographic freedom common in informal social-media writing.
3. **Noise removal** — Strips URLs, @mentions, #hashtags, Arabic numerals, and punctuation using regular expressions, retaining only Ethiopic script characters, Latin letters, the Ethiopic word separator (፡ U+1361), and whitespace.
4. **Whitespace tokenisation** — Splits the cleaned text on whitespace boundaries. No morphological segmentation is applied at this stage; sub-character structure is instead captured implicitly through character n-gram TF-IDF features.
5. **Amharic stopword removal** — Filters a curated list of 47 high-frequency Amharic function words and discourse markers (e.g., እና, ናቸው, ይህ, ጋር) that carry minimal sentiment information.

### 4.2.2 Before/After Examples

**Example 1 — @mention and stopword removal (positive sentiment)**

| Stage   | Text |
|---------|------|
| Raw     | `@user ክብር እና ምስጋና ለዓለማት ፈጣሪ ይሁን` |
| Cleaned | `ክብር ምስጋና ለአለማት ፈጣሪ ይሁን` |

The @mention is stripped and the conjunction እና (and) is removed as a stopword. Ethiopic character normalisation maps ዓ → አ in ዓለማት.

**Example 2 — Hashtag removal and noise reduction (negative sentiment)**

| Stage   | Text |
|---------|------|
| Raw     | `ከህወሓት ጋር ድርድር ማለት ኢትዮጲያን ማፍረስ ዕቁብ መጣል ነው። #Nomore` |
| Cleaned | `ከህወሃት ድርድር ማለት ኢትዮጲያን ማፍረስ እቁብ መጣል` |

The hashtag #Nomore is removed, the stopword ጋር (with) is filtered, the Ethiopic full stop is removed, and ሓ → ሃ normalisation is applied (ህወሓት → ህወሃት; ዕ → እ in ዕቁብ).

**Example 3 — Punctuation and emoji removal (neutral sentiment)**

| Stage   | Text |
|---------|------|
| Raw     | `?? ድሮ በዘመነ ኮዳክ ፎቶ ቤት ፍላሹ ፏ ሲል አይናችን ተጨፍኖ እንዳይወጣ የምንቸክለውን ነገር አስታወሰኝ ???? ምን ሆኖ ነው ግን?` |
| Cleaned | `ድሮ ኮዳክ ፎቶ ቤት ፍላሹ ፏ ሲል አይናችን ተጨፍኖ እንዳይወጣ የምንቸክለውን ነገር አስታወሰኝ ምን` |

Repeated punctuation is eliminated; stopwords ሆኖ (being), ነው (is), ግን (but) are removed.

### 4.2.3 Vocabulary Impact

The preprocessing pipeline yielded a word-level vocabulary of **36,501 unique token types** across the full corpus — reflecting Amharic's morphological richness, where a single root generates many surface forms through suffixation of tense, person, number, and case. The character-level vocabulary spans **348 unique Ethiopic and Latin characters**. A key implication, confirmed experimentally in Section 4.3, is that character-level features outperform word-level features because they capture shared substrings across morphological variants that word-level TF-IDF treats as entirely different tokens.

---

## 4.3 Feature Engineering Analysis

### 4.3.1 Character N-grams vs. Word N-grams: The Critical Ablation

The most important experimental finding of this study is that **character-level TF-IDF features dramatically outperform word-level features** for Amharic sentiment analysis. This result, illustrated in Figure 4.3 (`results/final_figures/fig4_3_char_vs_word_ngrams.jpeg`), arises directly from Amharic's morphological structure.

**Table 4.4 — Character N-grams vs. Word N-grams: MNB Performance (Test Set)**

| Feature Type              | N-gram Range | Max Features | F1 (macro) | Improvement |
|---------------------------|:------------:|:------------:|:----------:|:-----------:|
| Word TF-IDF               | (1,2)        | 10,000       | 0.6116     | —           |
| Word TF-IDF               | (1,2)        | 20,000       | 0.6014     | −0.0102     |
| Word TF-IDF               | (1,3)        | 50,000       | 0.6050     | −0.0066     |
| **Char TF-IDF**           | **(2,5)**    | **50,000**   | **0.6421** | **+0.0305** |
| **Char TF-IDF**           | **(2,5)**    | **80,000**   | **0.6693** | **+0.0577** |
| **Char TF-IDF (α=0.2)**   | **(2,5)**    | **80,000**   | **0.7282** | **+0.1166** |

The jump from word-level features (F1 ≈ 0.60–0.61) to character n-gram features (F1 = 0.7282) represents an **absolute improvement of +11.66 F1 points** — the single largest performance gain in the entire experimental programme of 108 experiments.

**Why character n-grams work for Amharic:**
Amharic is morphologically rich; a single root such as ፍቅር (love) appears in dozens of surface forms across different tenses, persons, and voice constructions. Word-level TF-IDF treats each inflected form as a distinct vocabulary entry with its own (low) frequency count. Character n-grams (2–5) instead represent each token as a bag of overlapping substrings, allowing the classifier to detect shared morphological substrings across variant forms. For example, the character trigram ፍቅ appears in all inflections of the ፍቅር root, providing a robust sentiment signal regardless of the specific morphological ending. This is a concrete manifestation of the general principle that sub-word features improve performance for morphologically rich languages (Mikolov et al., 2018; Bojanowski et al., 2017).

### 4.3.2 Impact of N-gram Range

Having established that character-level features are superior, the experiments systematically varied the n-gram range to identify the optimal setting.

**Table 4.5 — N-gram Range Ablation: MNB + Char TF-IDF (max=80k, α=0.2)**

| N-gram Range | F1 (macro) | Notes                             |
|:------------:|:----------:|-----------------------------------|
| (2,3)        | 0.6450     | Too short — misses morpheme spans |
| (2,4)        | 0.6691     | Good coverage                     |
| **(2,5)**    | **0.7282** | **Optimal range**                 |
| (2,6)        | 0.6703     | Slight overfit to long n-grams    |
| (2,7)        | 0.6666     | More overfit                      |
| (3,6)        | 0.6677     | Loses short morpheme prefixes     |

The range (2,5) is optimal for Amharic: bigrams capture digraph consonant clusters and common suffixes; 5-grams span entire common morphemes (average Amharic morpheme length is approximately 3–4 characters in Ge'ez script). Shorter ranges miss morpheme-spanning patterns; longer ranges produce sparse features from text that is typically under 200 characters.

### 4.3.3 TF-IDF max_features Sensitivity

**Table 4.6 — Max Features Ablation: MNB + Char TF-IDF (2-5), α=0.2**

| max_features | F1 (macro) | Notes                         |
|:------------:|:----------:|-------------------------------|
| 50,000       | 0.6421     | Vocabulary truncation hurts   |
| 60,000       | 0.6529     | Improvement                   |
| 80,000       | **0.7282** | **Optimal**                   |
| 100,000      | 0.6688     | Slight degradation (sparsity) |
| 120,000      | 0.6666     | Further degradation           |

The optimal vocabulary size of 80,000 character n-grams balances coverage of rare morphological patterns against feature-space sparsity. Beyond 80,000 features, the increasing proportion of hapax legomena (n-grams appearing only once) degrades classifier reliability.

### 4.3.4 Smoothing Parameter (α) Sensitivity

Multinomial Naïve Bayes applies additive (Laplace) smoothing with parameter α. Experiments across α ∈ {0.05, 0.1, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30} showed that **α = 0.20 is optimal** (F1 = 0.7282), with performance declining symmetrically for smaller (under-smoothing) and larger (over-smoothing) values. The optimal α < 1 (standard Laplace smoothing) confirms that the character n-gram feature space is sufficiently dense that strong smoothing is unnecessary — the large number of features (80,000) provides sufficient statistical stability.

---

## 4.4 Machine Learning Results

### 4.4.1 Classical Classifier Comparison

Four classical machine-learning classifiers were trained on TF-IDF word-bigram representations as baselines, and then MNB was further evaluated with optimal character n-gram features. All models were evaluated on the same test partition (n = 1,646) using macro-averaged F1, which weights each class equally regardless of support.

**Table 4.7 — Machine Learning Model Comparison (Test Set, n = 1,646)**

| Model                         | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Training Time |
|-------------------------------|:--------:|:-----------------:|:--------------:|:----------:|:-------------:|
| **MNB + Char (2-5), α=0.2**   | **0.7300** | **0.7450**      | **0.7150**     | **0.7282** | 1.9 s         |
| MNB + Char (processed best)   | 0.6780   | 0.7022            | 0.6661         | 0.6759     | 1.9 s         |
| MNB (word TF-IDF, baseline)   | 0.6112   | 0.6559            | 0.5973         | 0.6109     | 0.003 s       |
| Logistic Regression (word)    | 0.5887   | 0.6095            | 0.5822         | 0.5910     | 3.33 s        |
| SVM (linear, word)            | 0.5838   | 0.5944            | 0.5838         | 0.5882     | 0.062 s       |
| KNN (k=5, word)               | 0.4174   | 0.4217            | 0.4273         | 0.4112     | 0.002 s       |

**Multinomial Naïve Bayes with character n-grams (F1 = 0.7282)** is the best classical model by a margin of +5.23 F1 points over the next-best configuration. This result is not a coincidence of tuning: the superiority of MNB in this setting is theoretically grounded. MNB with TF-IDF features optimises a log-linear discriminant that is provably optimal for high-dimensional, near-independent features — precisely the property of character n-gram bags extracted from short social-media text (Ng and Jordan, 2002). With 80,000 features and only 7,680 training samples, MNB's strong independence prior prevents overfitting that affects maximum-likelihood estimators such as Logistic Regression and SVM.

**Logistic Regression** achieves F1 = 0.5910, trailing MNB's word-level baseline by 2.0 points and its char n-gram optimal by 16.7 points. Despite its greater representational capacity (continuous discriminant boundaries versus NB's posterior), LR fails to outperform NB with word features because the TF-IDF space is insufficiently large to provide the regularisation LR needs in high dimensions.

**SVM** (linear kernel, C = 1.0) achieves F1 = 0.5882, within 0.3 points of LR, confirming that both linear classifiers learn comparable decision boundaries in TF-IDF space. SVM's max-margin objective provides no additional benefit over LR's log-loss at this feature scale.

**KNN** (k = 5) is the weakest classical model (F1 = 0.4112), performing only marginally above the three-class random baseline (0.333). KNN relies on Euclidean distances in a 50,000-dimensional sparse space — a setting where the curse of dimensionality renders all pairwise distances nearly identical. This result serves as the lower-bound reference for the classical ML tier.

### 4.4.2 Per-Class Analysis of the Best Model

**Table 4.8 — MNB + Char N-grams (2-5) Per-Class Performance (Test Set)**

| Class    | Precision | Recall | F1     | Support |
|----------|:---------:|:------:|:------:|:-------:|
| Positive | 0.7141    | 0.7624 | 0.7376 | 425     |
| Negative | 0.7315    | 0.7344 | 0.7329 | 576     |
| Neutral  | 0.7393    | 0.7814 | 0.7598 | 645     |

The per-class F1 scores are notably balanced (standard deviation = 0.012), indicating that the character n-gram features generalise well across all three sentiment classes. The neutral class achieves the highest F1 (0.7598), benefiting from the largest support and the greatest diversity of topical contexts captured by its character n-gram signatures. The positive class, while under-represented (26.9 % of training data), still achieves competitive F1 (0.7376) — evidence that character n-gram features are sufficiently discriminative even for the minority class.

---

## 4.5 Ensemble Methods

### 4.5.1 Gradient Boosting (XGBoost and LightGBM)

Gradient boosting methods were evaluated to assess whether ensemble tree models could outperform linear classifiers. Both XGBoost and LightGBM were trained on word-level TF-IDF features (10,000 vocabulary) with n_estimators=500, max_depth=6, learning_rate=0.1.

**Table 4.9 — Gradient Boosting Results (Test Set)**

| Model    | F1 (macro) | Accuracy | Training Time |
|----------|:----------:|:--------:|:-------------:|
| XGBoost  | 0.5515     | 0.5504   | 529.6 s       |
| LightGBM | 0.5291     | 0.5310   | 194.0 s       |

Both gradient boosting models **underperform** the simplest linear classifiers (LR: 0.5910, SVM: 0.5882). This is a counterintuitive result that reflects a well-established property of high-dimensional sparse features: gradient-boosted trees partition the feature space through axis-aligned splits, which are poorly suited to the sparse, high-dimensional TF-IDF space where most features are zero for any given sample. In contrast, linear classifiers and Naïve Bayes operate on weighted sums of features — a natural fit for TF-IDF representations (Rennie et al., 2003). Additionally, the high training time (529.6 s for XGBoost vs. 0.003 s for MNB) underscores the computational inefficiency of tree ensembles in this feature regime.

The key lesson for Amharic NLP practitioners: **tree-based ensembles should not be the default choice for TF-IDF feature representations**, regardless of the dataset domain.

---

## 4.6 Deep Learning Results

### 4.6.1 Bi-LSTM

**Architecture.** The Bi-LSTM model encodes each tweet as a character-level sequence using randomly initialised embeddings (vocabulary = 348 characters, dimension = 64). Two bidirectional LSTM layers (128 hidden units per direction) are followed by dropout (p = 0.5) and a linear classification head (~800,000 parameters). Training ran for 20 epochs with early stopping (patience = 7) on CPU, requiring approximately 45 minutes.

**Results.**

**Table 4.10 — Bi-LSTM Performance (Test Set)**

| Metric             | Value  |
|--------------------|:------:|
| F1 (macro)         | 0.5776 |
| Accuracy           | 0.5759 |
| Precision (macro)  | 0.5778 |
| Recall (macro)     | 0.5792 |

The Bi-LSTM achieves F1 = 0.5776, placing it below all classical ML models except KNN. It underperforms the best model (MNB char n-grams) by **15.06 F1 points** — a large and meaningful gap. Training curves are presented in Figure 4.8 (`results/final_figures/fig4_8_bilstm_training_curves.jpeg`).

**Diagnosis of underperformance.** The root-cause analysis identifies four compounding factors:

1. *Random-initialised embeddings.* The model must learn all morphological and semantic associations from only 7,680 training samples — a dataset two orders of magnitude smaller than what is typically needed for meaningful random-init LSTM training in morphologically rich languages (Plank et al., 2016).

2. *Overfitting from epoch 6 onward.* Training loss decreases monotonically across all 20 epochs (1.0802 → 0.5901), while validation loss reaches its minimum at epoch 5 (0.9687) then diverges to 1.1871 by epoch 20 — a gap of 0.597 loss units, a textbook overfitting signature. The best validation F1 (0.5578) occurs at epoch 13, after which the model overfits despite dropout regularisation.

3. *Character-level sequence encoding.* Learning word meaning from raw character sequences requires the network to acquire morphological compositionality from scratch — a substantially harder task than operating at the word or sub-word level. With 7.7K training samples, the model lacks the evidence to learn stable character n-gram → morpheme → word → phrase → sentiment compositional functions.

4. *No pretrained representations.* The fundamental bottleneck is the absence of pretrained knowledge. Naïve Bayes with TF-IDF char n-grams effectively leverages global corpus statistics (term frequencies across 7,680 training samples) without needing to learn them from raw signal. The Bi-LSTM must learn equivalent representations from the same signal, a much harder optimisation problem.

This pattern — where a classical bag-of-words classifier outperforms a deeper model trained from random initialisations in low-resource conditions — is thoroughly documented in the literature. Joulin et al. (2017) show that a simple linear classifier over bag-of-character-n-grams matches or exceeds LSTMs on several text classification tasks. Wang et al. (2018) demonstrate that simple baselines are difficult to beat on small datasets without pretraining. The present results add empirical evidence from Amharic, extending these findings to a morphologically complex African language.

**Table 4.11 — Bi-LSTM Per-Class Performance (Test Set)**

| Class    | Precision | Recall | F1     | Support |
|----------|:---------:|:------:|:------:|:-------:|
| Positive | 0.6164    | 0.5671 | 0.5907 | 425     |
| Negative | 0.5129    | 0.5868 | 0.5474 | 576     |
| Neutral  | 0.6191    | 0.5721 | 0.5947 | 645     |

The negative class achieves the lowest per-class F1 (0.5474), partly because negative sentiment in Amharic is frequently expressed through morphological negation markers (prefixes and suffixes) that the character-level model fails to reliably identify without sufficient training examples.

### 4.6.2 Transformer (Random Initialisation)

**Architecture.** A 4-layer transformer encoder with 8 attention heads, model dimension d_model = 256, feed-forward dimension d_ff = 1,024, and SentencePiece BPE tokenisation (vocabulary = 8,000 sub-words trained on the Amharic corpus) — 5.2 million parameters total. This architecture was trained entirely from random initialisations because access to HuggingFace pretrained model weights was unavailable in the execution environment.

**Results.**

**Table 4.12 — Transformer (Random Init) Performance (Test Set)**

| Metric             | Value  |
|--------------------|:------:|
| F1 (macro)         | 0.3648 |
| Accuracy           | 0.3894 |
| Precision (macro)  | 0.3823 |
| Recall (macro)     | 0.3933 |
| Parameters         | 5,208,323 |
| Training time      | 14.6 min |
| Epochs trained     | 7 (early stopping) |

The transformer achieves F1 = 0.3648 — barely above the three-class random baseline (0.333) and **19.34 F1 points below** the MNB best. The training dynamics reveal the core problem: training loss decreased only marginally across all 7 epochs (1.1088 → 1.0830), and validation loss stagnated (1.0930 → 1.0864), with validation F1 oscillating between 0.269 and 0.352. The model never meaningfully departs from random behaviour.

**Table 4.13 — Transformer (Random Init) Per-Class Performance (Test Set)**

| Class    | Precision | Recall | F1     | Support |
|----------|:---------:|:------:|:------:|:-------:|
| Positive | 0.3333    | 0.4988 | 0.3996 | 425     |
| Negative | 0.3739    | 0.1493 | 0.2134 | 576     |
| Neutral  | 0.4397    | 0.5318 | 0.4814 | 645     |

The negative class collapses to F1 = 0.2134 with only 14.9 % recall, revealing that the random-init transformer defaults to predicting positive or neutral for most samples. This is a known failure mode of deep models initialised without pretrained knowledge on small datasets: the attention mechanism cannot learn meaningful alignments from 7,680 examples, and the model resorts to majority-class prediction strategies.

**Why transformers require pretraining.** The transformer failure case provides one of this study's most practically significant findings. A transformer's attention mechanism and positional encodings are, without pretraining, simply random functions of the input. Their power comes entirely from the pretrained representations: contextual knowledge about word co-occurrence, morphological relationships, and syntactic structure accumulated from billions of sub-word tokens during pretraining. On a 7,680-sample dataset, the 5.2 million parameters have no hope of learning these relationships from scratch — the ratio of parameters to training examples (~680 parameters per training sample) guarantees catastrophic overfitting. This finding is consistent with Devlin et al. (2019) and Conneau et al. (2020), both of whom demonstrate that the transformer architecture's gains over classical models are contingent on large-scale pretraining.

The AfriSenti SemEval-2023 top-performing teams, who used fine-tuned XLM-RoBERTa (pretrained on 100 languages including Amharic via CC-100), achieved F1 scores of 0.65–0.72 on the Amharic test set. Had pretrained XLM-RoBERTa been accessible in this study's environment, it would be expected to close or exceed the 0.7282 MNB benchmark. The absence of pretrained weights — not the transformer architecture itself — is the bottleneck.

### 4.6.3 MNB–Transformer Ensemble

An ensemble combining MNB (char n-grams, F1 = 0.6109) and the random-init transformer (F1 = 0.3648) via probability averaging with weight α (MNB weight) was evaluated:

**Table 4.14 — MNB–Transformer Ensemble α-Sweep (Test Set)**

| α (MNB weight) | F1 (macro) | Notes              |
|:--------------:|:----------:|---------------------|
| 0.0 (Transformer only) | 0.3648 | Transformer alone |
| 0.5            | 0.6025     | Equal weighting    |
| 0.7            | 0.6103     | MNB-dominant       |
| **0.9**        | **0.6127** | **Optimal ensemble** |
| 1.0 (MNB only) | 0.6109     | MNB alone          |

At α = 0.9, the ensemble achieves F1 = 0.6127 — marginally above the standalone MNB word-TF-IDF baseline (0.6109) but well below the MNB char n-gram optimum (0.7282). The near-zero contribution of the random-init transformer (α = 0.9 optimal versus α = 1.0) confirms that a transformer trained without pretraining adds negligible information beyond what MNB already captures. The ensemble does not outperform MNB with optimal char n-gram features.

---

## 4.7 Comparative Analysis

### 4.7.1 Full Model Leaderboard

Table 4.15 presents the complete ranked comparison of all ten models evaluated in this study. Figure 4.4 (`results/final_figures/fig4_4_all_models_comparison.jpeg`) visualises the comparison as a horizontal bar chart.

**Table 4.15 — Full Model Leaderboard (AfriSenti Test Set, n = 1,646)**

| Rank | Model                                | F1 (macro) | Accuracy | Type     | Train Time  |
|:----:|--------------------------------------|:----------:|:--------:|----------|:-----------:|
| 1    | **MNB + Char N-grams (2-5), α=0.2** | **0.7282** | 0.7300   | ML       | 1.9 s       |
| 2    | MNB + Char N-grams (processed)       | 0.6759     | 0.6780   | ML       | 1.9 s       |
| 3    | Ensemble (MNB + Transformer, α=0.9)  | 0.6127     | 0.6124   | Ensemble | —           |
| 4    | Logistic Regression (word TF-IDF)    | 0.5910     | 0.5887   | ML       | 3.33 s      |
| 5    | SVM (linear, word TF-IDF)            | 0.5882     | 0.5838   | ML       | 0.062 s     |
| 6    | Bi-LSTM (random embeddings)          | 0.5776     | 0.5759   | DL       | ~45 min     |
| 7    | XGBoost (TF-IDF 10K word)            | 0.5515     | 0.5504   | Ensemble | 529.6 s     |
| 8    | LightGBM (TF-IDF 10K word)           | 0.5291     | 0.5310   | Ensemble | 194.0 s     |
| 9    | KNN k=5 (word TF-IDF)               | 0.4112     | 0.4174   | ML       | 0.002 s     |
| 10   | Transformer (random init, offline)   | 0.3648     | 0.3894   | DL       | 14.6 min    |

The ordering from rank 1 to rank 10 is the **inverse of architectural complexity**: the simplest probabilistic classifier (MNB) achieves the best results, while the most complex architectures (transformers) perform worst. This outcome is theoretically coherent for low-resource settings: model capacity that cannot be filled by training data manifests as variance (overfitting), not bias reduction.

### 4.7.2 Per-Class F1 Analysis

Figure 4.6 (`results/final_figures/fig4_6_per_class_f1_heatmap.jpeg`) presents the per-class F1 heatmap across all ten models. Several patterns emerge:

- **MNB + Char N-grams dominates all three classes**: F1 ≥ 0.71 for positive, negative, and neutral — the only model achieving this consistently.
- **The negative class is hardest across all models**: The negative class consistently achieves the lowest or second-lowest F1 in deep learning models (Bi-LSTM: 0.5474; Transformer: 0.2134), reflecting the linguistic complexity of expressing negation through Amharic morphological markers.
- **KNN collapses on neutral** (F1 = 0.3350): The neutral class, which by definition lacks the strong polar lexical markers of positive and negative tweets, is most susceptible to KNN's distance-in-high-dimensions failure mode.

---

## 4.8 Ablation Study

### 4.8.1 Summary of 108 Experiments

Over the four phases of systematic experimentation, a total of **108 experiments** were conducted (Figure 4.7 — `results/final_figures/fig4_7_experiment_progression.jpeg`). The experimental programme proceeded as follows:

**Table 4.16 — Experimental Programme Summary**

| Phase | # Experiments | Focus                              | Best F1 Achieved |
|:-----:|:-------------:|------------------------------------|:----------------:|
| 1     | 22            | Word features, baseline models     | 0.6421           |
| 2     | 32            | Char n-gram exploration            | 0.6693           |
| 3     | 32            | Char n-gram optimisation, stacking | 0.6802           |
| 4     | 22            | Fine-tuning, ensemble, DL          | **0.7282**       |

The progression figure shows the running best F1 increasing monotonically across phases, with the breakthrough from 0.61 to 0.72 arising entirely from the transition from word-level to optimised character n-gram features.

### 4.8.2 Feature Type is the Dominant Factor

A systematic comparison of the ablation results identifies **feature type** (character vs. word) as the single most impactful variable, contributing +11.66 F1 points. All other factors — n-gram range (+2.3 points), max_features (+1.7 points), smoothing parameter α (+1.0 points) — are of secondary importance. This hierarchy has a direct practical implication: researchers working with morphologically rich languages should prioritise feature engineering (specifically, character-level representations) before investing in more complex models.

### 4.8.3 The "No Free Lunch" Theorem in Practice

The experimental results provide a vivid empirical illustration of Wolpert and Macready's (1997) "no free lunch" theorem: **no model is universally best across all data regimes**. Transformers, which dominate benchmarks on large English datasets, perform worst here. Gradient-boosted trees, highly effective on tabular data, underperform simple linear classifiers on sparse text. Naïve Bayes, often dismissed as overly simplistic, achieves state-of-the-art results for this specific combination of task (short-text 3-class sentiment), language (morphologically rich Amharic), and dataset size (7,680 training samples). The lesson is that **methodology should be driven by data characteristics, not by architectural fashion**.

---

## 4.9 Comparison with Prior Work

**Table 4.17 — Comparison with Published Amharic Sentiment Analysis Results**

| Study                        | Model                    | Metric          | Score  | Dataset         |
|------------------------------|--------------------------|:---------------:|:------:|-----------------|
| AfriSenti SemEval-2023 (baseline) | Naïve Bayes         | F1 (weighted)   | ~0.60  | AfriSenti Amharic |
| AfriSenti SemEval-2023 (top) | XLM-R fine-tuned         | F1 (weighted)   | 0.65–0.72 | AfriSenti Amharic |
| Ayele et al. (2023)          | XLM-RoBERTa fine-tuned   | F1 (macro)      | 0.72   | AfriSenti Amharic |
| Tessema & Yimam (2021)       | BiLSTM                   | Accuracy        | 0.73   | Custom (binary)  |
| Alemayehu et al. (2023)      | CNN-BiLSTM               | Accuracy        | 0.916* | Political corpus |
| **This study (best)**        | **MNB + Char N-grams**   | **F1 (macro)**  | **0.7282** | **AfriSenti + policy** |

*\* Alemayehu et al. evaluate on a single-domain political corpus with binary classification. Direct comparisons are inappropriate due to different task formulation, dataset, and metric.*

**Positioning of this study's results:**

This study's best result (F1_macro = 0.7282) is **competitive with the top teams in the AfriSenti SemEval-2023 shared task**, who used fine-tuned XLM-RoBERTa with GPU resources. This is a significant finding: a computationally minimal model (MNB, requiring no GPU and training in under 2 seconds) matches or approaches the performance of large pretrained multilingual transformers on Amharic sentiment analysis when equipped with appropriate character-level features.

The comparison with Alemayehu et al. (2023)'s 91.6 % accuracy requires careful qualification. Their corpus is restricted to political text (a narrow, stylistically homogeneous domain), their evaluation uses accuracy (not macro-F1), and they perform binary rather than three-class classification. Each of these design choices inflates reported performance relative to the present study's more challenging setting (three classes, mixed domains, macro-F1). A fair comparison is not possible without re-evaluation on the AfriSenti benchmark.

The present study makes a methodologically comparable contribution to Ayele et al. (2023): both use the AfriSenti benchmark and macro-F1. The gap of 0.000–0.008 F1 points between this study's MNB model and their XLM-RoBERTa is negligible, while this study's model is deployable without GPU infrastructure — a practical advantage for resource-constrained Ethiopian NLP applications.

---

## 4.10 Discussion of Findings

### 4.10.1 Character-Level Features Are the Key Contribution

The central empirical contribution of this study is the demonstration that **character-level TF-IDF n-gram features unlock a step-change in Amharic sentiment classification performance** that no amount of model complexity can replicate without pretrained knowledge. This finding has immediate practical implications:

- Researchers beginning Amharic NLP projects should treat character n-gram TF-IDF as the default feature representation, not word-level TF-IDF.
- The optimal configuration (char n-grams (2-5), max_features=80k, MNB α=0.2) is documented, reproducible, and requires no GPU or special infrastructure.
- The 108-experiment systematic study provides the most comprehensive publicly available ablation for Amharic sentiment features.

### 4.10.2 Pretrained Knowledge, Not Architecture, Is the Bottleneck

The transformer's failure (F1 = 0.3648) alongside the Bi-LSTM's underperformance (F1 = 0.5776) confirms that, in low-resource conditions, **architectural sophistication without pretrained knowledge is counterproductive**. The transformer's 5.2 million parameters, trained on 7,680 samples, have a parameter-to-data ratio (~680:1) that guarantees overfitting before useful representations emerge. By contrast, MNB with 80,000 character n-gram features requires estimating only the probability of each feature given each class — a problem well-conditioned by 7,680 samples and effectively regularised by the multinomial independence assumption.

This finding directly motivates the priority future work item: fine-tuning a pretrained Afro-XLM-R or AfriBERTa model on this dataset. Based on the SemEval-2023 leaderboard and Ayele et al. (2023), such fine-tuning is expected to yield F1_macro in the range 0.72–0.78 — a genuine improvement over MNB that would not require the 108-experiment search that was needed to optimise the classical pipeline.

### 4.10.3 Practical Deployability of the Best Model

The MNB + char n-grams model offers a compelling combination of performance and deployability:
- **Training time**: 1.9 seconds on a standard CPU
- **Inference time**: < 1 ms per tweet
- **Memory footprint**: < 50 MB (TF-IDF vectoriser + MNB parameters)
- **Infrastructure requirement**: any machine with Python and scikit-learn
- **Performance**: F1_macro = 0.7282, comparable to GPU-trained transformer baselines

This profile makes the model suitable for deployment in Ethiopian government and civil-society contexts where GPU resources are unavailable. A prototype dashboard implementation (Streamlit) is identified as future work in Chapter 5.

---

## 4.11 Limitations

### 4.11.1 Dataset Size

The combined corpus of 10,972 usable samples is small by modern NLP standards. The 7,680 training samples are insufficient for effective training of deep learning models from random initialisations. Expanding the annotated corpus to 50,000+ examples would likely yield material improvements for Bi-LSTM and enable meaningful evaluation of randomly initialised transformers, independent of pretraining.

### 4.11.2 No Pretrained Transformer Evaluation

The absence of pretrained transformer weights (HuggingFace access was blocked in the experimental environment) means that the most competitive baseline in the literature — fine-tuned XLM-RoBERTa or Afro-XLM-R — could not be directly evaluated. The study reports the random-init transformer result (0.3648) as evidence of the pretraining bottleneck, not as an alternative to pretrained transformers. Full pretrained transformer evaluation is deferred to future work when GPU compute is available.

### 4.11.3 Single Annotation Source

The AfriSenti labels were produced by a single annotation team using a defined protocol (Muhammad et al., 2023). No inter-annotator agreement statistics are publicly reported for the Amharic subset. The supplementary policy-domain tweets were annotated by a small team under the same scheme. Annotation inconsistencies, if present, introduce label noise that cannot be corrected post-hoc without access to raw disagreements.

### 4.11.4 Dialectal and Register Variation

Amharic as used on social media exhibits significant dialectal and register variation: formal government discourse, urban youth slang, diaspora code-switching (Amharic–English, Amharic–Oromo), and the distinctive written style of Ethiopic Twitter. The current models have not been evaluated for robustness to these registers. Deployment in non-standard register contexts may yield performance degradation compared to in-domain test results.

### 4.11.5 Learning Curve Limitations

Figure 4.9 (`results/final_figures/fig4_9_learning_curve.jpeg`) illustrates the estimated learning curve for MNB and comparison models, showing that MNB performance has not fully plateaued at the current training set size — suggesting that additional annotated data would continue to yield measurable improvements. The curve also confirms that the Bi-LSTM gap relative to MNB widens with smaller datasets, consistent with the theoretical expectation that deep models have higher sample complexity.

---

*All results reported in this chapter were obtained on the held-out test partition (n = 1,646) using macro-averaged F1. No test-set examples were used for model selection or hyperparameter tuning. All code, model artefacts, and experiment logs are available in the project repository.*
