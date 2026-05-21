"""Chapter 3 — Research Methodology."""

from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from build_thesis import (body_para, chapter_title, section_heading,
                           add_run, make_table, page_break, add_figure, FIG)

def build_chapter3(doc):
    chapter_title(doc, "CHAPTER THREE")
    chapter_title(doc, "RESEARCH METHODOLOGY")

    body_para(doc, (
        "This chapter describes the research design, dataset, preprocessing pipeline, "
        "feature extraction approaches, model architectures, evaluation metrics, "
        "experimental setup, and ethical considerations that governed the study. "
        "All steps described here were completed before the results reported in "
        "Chapter Four were obtained; the chapter is written in past tense to reflect "
        "the completed nature of the work."), first_indent=False)

    # 3.1
    section_heading(doc, "3.1 Research Design")
    body_para(doc, (
        "This study employed an experimental research design with systematic model "
        "comparison. The central approach was a controlled ablation study in which "
        "a common dataset, preprocessing pipeline, and evaluation metric were held "
        "constant while the model architecture and feature representation were varied "
        "systematically across 108 experiments. This design enabled causal attribution "
        "of performance differences to specific modelling choices rather than "
        "confounding factors such as data quality or evaluation methodology."))
    body_para(doc, (
        "The experimental design proceeded through four phases: baseline ML experiments "
        "with word-level features; character n-gram exploration; character n-gram "
        "optimisation and stacking; and fine-tuning, ensemble methods, and deep learning. "
        "Each phase built on the findings of the previous, implementing an iterative "
        "refinement strategy consistent with established machine learning research practice."))

    # 3.2
    section_heading(doc, "3.2 Dataset Description")
    section_heading(doc, "3.2.1 AfriSenti Amharic Dataset", level=2)
    body_para(doc, (
        "The primary data source was the AfriSenti Amharic benchmark dataset (Muhammad "
        "et al., 2023), a publicly available collection of Amharic-language tweets "
        "annotated for three-class sentiment (positive, negative, neutral). The dataset "
        "was released as part of the SemEval-2023 Task 12 shared task and provides "
        "official train, development, and test splits with standardised annotation "
        "guidelines developed by the AfriSenti consortium. Approximately 8,950 tweets "
        "were drawn from this source."))
    body_para(doc, (
        "A supplementary component of 2,530 Amharic tweets was collected from Twitter/X "
        "using Ethiopian government-policy keywords spanning the domains of education, "
        "health, economy, and security. These tweets were annotated under the same "
        "three-class scheme as AfriSenti by a team of native Amharic speakers. The "
        "combined corpus totalled 11,477 Amharic-language tweets before preprocessing."))

    section_heading(doc, "3.2.2 Dataset Characteristics", level=2)
    body_para(doc, (
        "After applying the preprocessing pipeline described in Section 3.3, 10,972 "
        "samples were retained for modelling (505 samples were discarded as empty after "
        "preprocessing). These were partitioned using stratified random splitting with "
        "random seed 42 to maintain class proportions across all three sets."))

    ds_headers = ["Split", "Samples", "Percentage"]
    ds_rows = [
        ["Training", "7,680", "70.0%"],
        ["Validation", "1,646", "15.0%"],
        ["Test", "1,646", "15.0%"],
        ["Total", "10,972", "100%"],
    ]
    make_table(doc, ds_headers, ds_rows, "Table 3.1: Dataset Statistics — AfriSenti Amharic Corpus")

    body_para(doc, (
        "The corpus exhibited moderate class imbalance: the negative and neutral classes "
        "together accounted for 73.1% of all samples, with the positive class comprising "
        "only 26.9%. This imbalance was addressed through stratified splitting, which "
        "preserved class proportions in each partition, and through the use of macro-"
        "averaged F1 as the primary evaluation metric, which weights all classes equally "
        "regardless of support."))

    # 3.3
    section_heading(doc, "3.3 Data Preprocessing")
    body_para(doc, (
        "The preprocessing pipeline comprised five sequential stages applied to every "
        "tweet before feature extraction. Each stage addressed a specific source of "
        "noise or inconsistency characteristic of informal Amharic social media text."))

    steps = [
        ("Unicode NFC Normalisation",
         "All text was converted to Unicode Canonical Decomposition, followed by Canonical "
         "Composition (NFC) normalisation, ensuring consistent byte representations for "
         "Ethiopic code points. Invisible combining characters introduced by inconsistent "
         "keyboard encodings were removed at this stage."),
        ("Ethiopic Character Normalisation",
         "Phonetically equivalent Ge'ez character variants were mapped to canonical forms. "
         "The ሐ/ኀ groups were mapped to ሃ; the ዐ group was mapped to አ; and the ፀ group "
         "was mapped to the ጸ group. This normalisation reduced vocabulary fragmentation "
         "caused by the orthographic freedom common in informal Amharic writing."),
        ("Noise Removal",
         "URLs, @mentions, #hashtags, Arabic numerals, and punctuation were stripped using "
         "regular expressions. Only Ethiopic script characters, Latin letters, the Ethiopic "
         "word separator (፡ U+1361), and whitespace were retained."),
        ("Whitespace Tokenisation",
         "The cleaned text was split on whitespace boundaries. No morphological segmentation "
         "was applied at this stage; sub-character structure was captured implicitly through "
         "character n-gram TF-IDF features in the feature extraction stage."),
        ("Amharic Stopword Removal",
         "A curated list of 47 high-frequency Amharic function words and discourse markers "
         "(e.g., እና, ናቸው, ይህ, ጋር) that carry minimal sentiment information was filtered "
         "from the tokenised text."),
    ]
    for i, (name, desc) in enumerate(steps, 1):
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.left_indent = Inches(0.5)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
        p.paragraph_format.line_spacing = 1.5
        add_run(p, f"{i}. {name}: ", bold=True, size=12)
        add_run(p, desc, size=12)

    body_para(doc, (
        "The preprocessing pipeline yielded a word-level vocabulary of 36,501 unique token "
        "types across the full corpus, reflecting Amharic's morphological richness. The "
        "character-level vocabulary spanned 348 unique Ethiopic and Latin characters. The "
        "reduction of vocabulary fragmentation through Ethiopic character normalisation was "
        "the single most impactful preprocessing step for character n-gram feature quality."))

    # 3.4
    section_heading(doc, "3.4 Feature Extraction")
    section_heading(doc, "3.4.1 TF-IDF with Word N-grams", level=2)
    body_para(doc, (
        "Word-level TF-IDF vectorisation was used as the baseline feature representation. "
        "The TF-IDF weight for term t in document d is defined as:"))
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(6)
    add_run(p, "TF-IDF(t, d) = TF(t, d) × log(N / DF(t))", italic=True, size=12)

    body_para(doc, (
        "where TF(t, d) is the frequency of term t in document d, N is the total number "
        "of documents, and DF(t) is the number of documents containing term t. Word "
        "n-gram configurations explored included unigrams (1,1), bigrams (1,2), and "
        "trigrams (1,3) with vocabulary sizes of 10,000 to 50,000."))

    section_heading(doc, "3.4.2 TF-IDF with Character N-grams", level=2)
    body_para(doc, (
        "Character-level TF-IDF vectorisation applied the same TF-IDF weighting formula "
        "to character n-grams — overlapping substrings of n consecutive characters — "
        "rather than word tokens. This approach provides implicit morphological smoothing "
        "for Amharic: character n-grams capture shared substrings across morphological "
        "variants of the same root, allowing the classifier to detect sentiment signals "
        "even for rare or previously unseen inflected forms."))
    body_para(doc, (
        "The optimal configuration identified through systematic ablation was character "
        "n-grams with range (2,5) and max_features=80,000. The range (2,5) covers "
        "character bigrams through 5-grams, spanning the typical length of Amharic "
        "morphemes (3–4 Ge'ez characters on average). The vocabulary size of 80,000 "
        "balanced coverage of morphological patterns against feature-space sparsity."))

    section_heading(doc, "3.4.3 Sub-word Tokenisation for Transformer Experiments", level=2)
    body_para(doc, (
        "For the transformer model, SentencePiece Byte Pair Encoding (BPE) tokenisation "
        "was trained on the Amharic corpus with a vocabulary size of 8,000 sub-word "
        "units. BPE iteratively merges the most frequent character pair in the corpus, "
        "producing a vocabulary of common character sequences that balances coverage "
        "of frequent words with efficient encoding of rare morphological forms. The "
        "BPE tokeniser was trained exclusively on the training partition to prevent "
        "data leakage."))

    # 3.5
    section_heading(doc, "3.5 Model Architectures")
    section_heading(doc, "3.5.1 Multinomial Naïve Bayes", level=2)
    body_para(doc, (
        "Multinomial Naïve Bayes applies Bayes' theorem under the conditional independence "
        "assumption to classify documents based on feature count vectors. The classifier "
        "estimates the probability of class c given document d as: "
        "P(c|d) ∝ P(c) × ∏ P(f|c)^count(f,d), "
        "where the product runs over all features f in d. Laplace additive smoothing "
        "with parameter α prevents zero-probability estimates for unseen features. "
        "The optimal α=0.20 was identified through systematic grid search (Section 4.3.4)."))

    section_heading(doc, "3.5.2 Logistic Regression", level=2)
    body_para(doc, (
        "Logistic Regression was implemented with L2 regularisation (C=1.0) using "
        "the saga solver for multi-class classification via the one-vs-rest strategy. "
        "The model learned a continuous log-linear discriminant boundary in the TF-IDF "
        "feature space. Word-level TF-IDF bigrams (max_features=50,000) were used "
        "as the feature representation following baseline experiments."))

    section_heading(doc, "3.5.3 Support Vector Machine", level=2)
    body_para(doc, (
        "An SVM with a linear kernel was trained using the LinearSVC implementation "
        "with L2 regularisation (C=1.0) and a maximum of 1,000 iterations. The linear "
        "kernel was selected based on established results showing that high-dimensional "
        "TF-IDF spaces are already approximately linearly separable for text classification."))

    section_heading(doc, "3.5.4 K-Nearest Neighbours", level=2)
    body_para(doc, (
        "KNN was implemented with k=5 neighbours, Euclidean distance metric, and uniform "
        "weighting. Word-level TF-IDF features (max_features=50,000) were used. "
        "KNN served as the lower-bound baseline for classical ML performance, providing "
        "empirical evidence of the curse of dimensionality in high-dimensional sparse spaces."))

    section_heading(doc, "3.5.5 XGBoost", level=2)
    body_para(doc, (
        "XGBoost was trained with n_estimators=500, max_depth=6, learning_rate=0.1, "
        "and a word-level TF-IDF vocabulary of 10,000 features. Early stopping was "
        "applied based on the validation set. XGBoost was evaluated to assess whether "
        "gradient-boosted tree ensembles could outperform linear classifiers on the "
        "high-dimensional sparse TF-IDF representation."))

    section_heading(doc, "3.5.6 LightGBM", level=2)
    body_para(doc, (
        "LightGBM was trained with equivalent hyperparameters to XGBoost "
        "(n_estimators=500, max_depth=6, learning_rate=0.1) on word-level TF-IDF "
        "features (max_features=10,000). LightGBM's leaf-wise tree growth strategy "
        "provided faster training than XGBoost's level-wise approach while producing "
        "comparable results."))

    section_heading(doc, "3.5.7 Bidirectional LSTM", level=2)
    body_para(doc, (
        "The Bi-LSTM model encoded each tweet as a character-level sequence using "
        "randomly initialised embeddings (vocabulary=348 characters, dimension=64). "
        "Two bidirectional LSTM layers with 128 hidden units per direction were "
        "followed by dropout (p=0.5) and a linear classification head. The total "
        "parameter count was approximately 800,000. Training ran for up to 20 epochs "
        "with early stopping (patience=7) on the validation macro-F1. The Adam "
        "optimiser was used with a learning rate of 1e-3."))

    section_heading(doc, "3.5.8 Transformer (Random Initialisation)", level=2)
    body_para(doc, (
        "A 4-layer transformer encoder was implemented with 8 attention heads, model "
        "dimension d_model=256, feed-forward dimension d_ff=1,024, and SentencePiece "
        "BPE tokenisation (vocabulary=8,000 sub-words). The total parameter count was "
        "5,208,323. This model was trained entirely from random initialisations because "
        "access to HuggingFace pretrained model weights was unavailable in the execution "
        "environment. Training used the AdamW optimiser with learning rate 5e-4 and "
        "cosine learning rate schedule, with early stopping (patience=3) on "
        "validation macro-F1."))

    section_heading(doc, "3.5.9 MNB–Transformer Ensemble", level=2)
    body_para(doc, (
        "An ensemble combining MNB (char n-grams, word-TF-IDF configuration) and "
        "the random-init transformer was evaluated via soft probability averaging: "
        "P_ensemble = α × P_MNB + (1-α) × P_Transformer, "
        "where α was varied over the range [0.0, 1.0] in steps of 0.1. The optimal "
        "α was identified empirically on the validation set."))

    # Hyperparameter table
    hp_headers = ["Model", "Key Hyperparameters", "Features"]
    hp_rows = [
        ["MNB (best)", "α=0.20", "Char TF-IDF (2-5), max=80k"],
        ["Logistic Regression", "C=1.0, L2, solver=saga", "Word TF-IDF (1-2), max=50k"],
        ["SVM", "C=1.0, linear kernel", "Word TF-IDF (1-2), max=50k"],
        ["KNN", "k=5, Euclidean", "Word TF-IDF (1-2), max=50k"],
        ["XGBoost", "n_est=500, depth=6, lr=0.1", "Word TF-IDF (1-2), max=10k"],
        ["LightGBM", "n_est=500, depth=6, lr=0.1", "Word TF-IDF (1-2), max=10k"],
        ["Bi-LSTM", "128 units×2, dropout=0.5, lr=1e-3", "Char embeddings (dim=64)"],
        ["Transformer", "4 layers, 8 heads, d=256, lr=5e-4", "BPE (vocab=8k)"],
        ["Ensemble", "α=0.9 (MNB weight)", "MNB char + Transformer BPE"],
    ]
    make_table(doc, hp_headers, hp_rows, "Table 3.2: Hyperparameter Configurations for All Models")

    # 3.6
    section_heading(doc, "3.6 Evaluation Metrics")
    body_para(doc, (
        "All models were evaluated using a consistent set of classification metrics "
        "computed on the held-out test partition (n=1,646). The primary metric was "
        "macro-averaged F1-score, with accuracy, precision, and recall reported "
        "as supporting metrics."))

    body_para(doc, ("The key metrics are defined as follows:"), first_indent=False)
    metrics_defs = [
        ("Precision (per class)", "TP / (TP + FP)"),
        ("Recall (per class)", "TP / (TP + FN)"),
        ("F1-score (per class)", "2 × (Precision × Recall) / (Precision + Recall)"),
        ("Macro-averaged F1", "(1/K) × Σ F1_k  for k = 1..K classes"),
        ("Accuracy", "(TP + TN) / (TP + TN + FP + FN)"),
    ]
    for metric, formula in metrics_defs:
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.left_indent = Inches(0.5)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
        p.paragraph_format.line_spacing = 1.5
        add_run(p, f"{metric}: ", bold=True, size=12)
        add_run(p, formula, italic=True, size=12)

    body_para(doc, (
        "Macro-averaged F1 was chosen as the primary metric because it weights all "
        "three sentiment classes equally regardless of their support in the test set. "
        "This is the appropriate metric for an imbalanced corpus where the positive "
        "class is under-represented (26.9% of samples): accuracy would overweight "
        "the majority classes and produce misleadingly optimistic results for "
        "classifiers that systematically misclassify the minority class. Macro-F1 "
        "was also the metric used by the AfriSenti SemEval-2023 shared task, enabling "
        "direct comparison with published results."))

    # 3.7
    section_heading(doc, "3.7 Experimental Setup")
    body_para(doc, (
        "All experiments were conducted in a CPU-only environment (Intel x86-64, "
        "no GPU acceleration) to reflect the computational constraints of Ethiopian "
        "institutional settings. The software stack comprised Python 3.10, scikit-learn "
        "1.3, PyTorch 2.0, pandas 2.0, and SentencePiece 0.1.99. All experiments used "
        "random seed 42 for reproducibility. The 108 experiments were organised as a "
        "systematic grid search across model type, feature type, n-gram range, "
        "vocabulary size, and smoothing parameters, with each experiment logged to "
        "a CSV file (results/experiments/experiment_log.csv) with full hyperparameter "
        "specifications and test-set metrics."))

    # 3.8
    section_heading(doc, "3.8 Ethical Considerations")
    body_para(doc, (
        "This study used the AfriSenti dataset, a publicly available benchmark released "
        "under its standard academic-use terms (Muhammad et al., 2023). The supplementary "
        "policy-domain tweets were collected using Twitter/X's public API under its "
        "developer terms of service, which permit academic research. No personally "
        "identifiable information beyond what is publicly available was stored; tweet "
        "text was retained but user identifiers were not. The annotation of supplementary "
        "tweets followed the same three-class protocol as AfriSenti, with annotators "
        "informed of the academic purpose of the annotation task."))
    body_para(doc, (
        "Ethical approval was obtained from the researcher's institution prior to data "
        "collection. The study does not involve human subjects experimentation; it "
        "analyses publicly available text data. The potential harms of Amharic sentiment "
        "analysis tools — including surveillance of political speech or amplification "
        "of biased sentiment classifications — are acknowledged and discussed in the "
        "limitations and recommendations sections. The open-source release of the "
        "pipeline supports transparency and reproducibility."))

    # Figure 3.1
    doc.add_paragraph()
    add_figure(doc,
               f"{FIG}/fig3_1_methodology_pipeline.jpeg",
               "Figure 3.1: Research Methodology Pipeline",
               width=5.5)
    page_break(doc)

print("Chapter 3 function defined.")
