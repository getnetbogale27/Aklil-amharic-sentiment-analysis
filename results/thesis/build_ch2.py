"""Chapter 2 — Review of Related Literature."""

from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from build_thesis import (body_para, chapter_title, section_heading,
                           add_run, make_table, page_break)

def build_chapter2(doc):
    chapter_title(doc, "CHAPTER TWO")
    chapter_title(doc, "REVIEW OF RELATED LITERATURE")

    body_para(doc, (
        "This chapter reviews the theoretical foundations and empirical research relevant to "
        "the development of an AI-driven Amharic sentiment analysis framework for public policy "
        "enhancement. The review is organised into four sections: theoretical literature, "
        "empirical literature, the conceptual framework, and the research gap that this "
        "study addresses."), first_indent=False)

    # ── 2.1 Theoretical ──────────────────────────────────────────────────────
    section_heading(doc, "2.1 Theoretical Literature Review")

    section_heading(doc, "2.1.1 Natural Language Processing and Sentiment Analysis", level=2)
    body_para(doc, (
        "Natural Language Processing (NLP) is a multidisciplinary field at the intersection of "
        "linguistics, computer science, and artificial intelligence that enables machines to "
        "understand, interpret, and generate human language. NLP tasks include text classification, "
        "machine translation, named entity recognition, question answering, and sentiment analysis "
        "(Pang & Lee, 2008). The field has undergone three major paradigm shifts: rule-based "
        "systems (1950s–1990s), statistical and machine learning approaches (1990s–2010s), "
        "and deep learning and transformer-based methods (2013–present)."))
    body_para(doc, (
        "Sentiment analysis, the focus of this study, is a text classification task in which "
        "each input document is assigned to one of a predefined set of sentiment classes. The "
        "most common formulation is three-class classification: positive, negative, and neutral "
        "(Liu, 2022). Three primary approaches have been developed. Lexicon-based approaches "
        "rely on sentiment dictionaries that assign polarity scores to words; they require no "
        "annotated training data but suffer from coverage limitations and sensitivity to "
        "negation and contextual modification. Machine learning approaches train discriminative "
        "classifiers on labelled corpora, learning the statistical association between textual "
        "features (n-grams, TF-IDF weights) and sentiment labels. Deep learning approaches "
        "train neural networks to learn feature representations directly from raw text, "
        "potentially capturing long-range dependencies and morphological patterns that "
        "hand-engineered features miss."))
    body_para(doc, (
        "For morphologically rich, low-resource languages such as Amharic, the choice of "
        "approach is constrained by data availability. Lexicon-based approaches require "
        "sentiment lexicons that do not yet exist for Amharic at sufficient scale. Deep "
        "learning approaches require large training corpora and, for transformers, large-scale "
        "pretraining data. Machine learning approaches with carefully engineered features "
        "represent the practical optimum when labelled data is limited to the thousands "
        "rather than millions of examples."))

    section_heading(doc, "2.1.2 Machine Learning Approaches to Text Classification", level=2)
    body_para(doc, (
        "Classical machine learning approaches to text classification operate on a "
        "bag-of-features representation, most commonly TF-IDF (Term Frequency-Inverse "
        "Document Frequency) weighted n-gram vectors. TF-IDF assigns each term a weight "
        "proportional to its frequency in the document and inversely proportional to "
        "its frequency across the corpus, effectively emphasising discriminative terms "
        "over common function words."))
    body_para(doc, (
        "Multinomial Naïve Bayes (MNB) applies Bayes' theorem under the conditional "
        "independence assumption, estimating the probability of each class given the "
        "observed feature counts. Despite its simplifying assumption, MNB is theoretically "
        "well-suited to high-dimensional, near-independent features — precisely the property "
        "of n-gram TF-IDF bags from short social-media text (Ng & Jordan, 2002). Logistic "
        "Regression learns a log-linear discriminant boundary through maximum-likelihood "
        "estimation with L2 regularisation. Support Vector Machines (SVM) find the maximum-"
        "margin hyperplane separating classes in the feature space; with a linear kernel, "
        "SVM is equivalent to a regularised logistic regression under certain conditions. "
        "K-Nearest Neighbour (KNN) classifies based on majority vote among the k most "
        "similar training examples, where similarity is measured by Euclidean distance in "
        "the feature space — a strategy prone to failure in high-dimensional sparse spaces "
        "(the curse of dimensionality)."))
    body_para(doc, (
        "Gradient-boosted tree ensembles (XGBoost, LightGBM) have achieved state-of-the-art "
        "results on tabular data but are theoretically ill-suited to high-dimensional sparse "
        "TF-IDF vectors, where axis-aligned tree splits cannot efficiently partition the "
        "feature space (Rennie et al., 2003). This theoretical expectation is confirmed "
        "experimentally in Chapter 4 of this study."))

    section_heading(doc, "2.1.3 Deep Learning for NLP", level=2)
    body_para(doc, (
        "Deep learning approaches learn hierarchical feature representations directly from "
        "raw text. Convolutional Neural Networks (CNNs) apply learned filters to sequences "
        "of word or character embeddings, capturing local n-gram patterns without assuming "
        "sequential dependencies (Kim, 2014). Long Short-Term Memory networks (LSTMs) "
        "are a type of recurrent neural network designed to capture long-range sequential "
        "dependencies through gating mechanisms that regulate information flow (Hochreiter "
        "& Schmidhuber, 1997). Bidirectional LSTMs (Bi-LSTMs) extend LSTMs by processing "
        "each sequence in both forward and backward directions, producing contextualised "
        "representations that incorporate both left and right context for each token."))
    body_para(doc, (
        "The fundamental limitation of deep learning models in low-resource settings is "
        "sample complexity: the number of training examples required for the model to "
        "learn useful representations from random initialisations scales with the number "
        "of parameters. A Bi-LSTM with 800,000 parameters trained on 7,680 examples faces "
        "an unfavourable parameter-to-data ratio that leads to overfitting. Dropout "
        "regularisation (Srivastava et al., 2014) partially mitigates this risk but "
        "cannot compensate for the fundamental absence of pretrained semantic knowledge."))

    section_heading(doc, "2.1.4 Transformer Models and Transfer Learning", level=2)
    body_para(doc, (
        "The transformer architecture, introduced by Vaswani et al. (2017), replaced "
        "recurrence with self-attention mechanisms that compute representations of each "
        "token by attending to all other tokens in the sequence simultaneously. This "
        "parallelisable architecture enabled scaling to models with billions of parameters "
        "trained on massive multilingual corpora."))
    body_para(doc, (
        "BERT (Devlin et al., 2019) demonstrated that a bidirectional transformer pretrained "
        "on masked language modelling and next-sentence prediction tasks produces universal "
        "representations that can be fine-tuned for downstream NLP tasks with minimal "
        "task-specific architecture. XLM-RoBERTa (Conneau et al., 2020) extended BERT to "
        "100 languages, including Amharic, through pretraining on the CC-100 multilingual "
        "corpus. Afro-XLM-R (Alabi et al., 2022) further adapted XLM-RoBERTa to African "
        "languages through multilingual adaptive fine-tuning on African text corpora, "
        "demonstrating consistent improvements over vanilla XLM-RoBERTa on African NLP tasks "
        "including the AfriSenti SemEval-2023 shared task."))
    body_para(doc, (
        "AfriBERTa (Ogueji et al., 2021) is a BERT-based model trained from scratch on "
        "text from eleven African languages including Amharic, Hausa, Yoruba, and Swahili. "
        "Its Amharic pretraining data, while smaller than XLM-RoBERTa's multilingual corpus, "
        "is more targeted and may provide better inductive bias for Amharic morphological "
        "structure. Both Afro-XLM-R and AfriBERTa represent the frontier for pretrained "
        "Amharic NLP and are identified as priority future work in this study."))

    section_heading(doc, "2.1.5 Feature Extraction for Morphologically Rich Languages", level=2)
    body_para(doc, (
        "Amharic's agglutinative morphology means that a single lexical root generates "
        "dozens of inflected surface forms through affixation. Word-level tokenisation "
        "treats each inflected form as a distinct vocabulary entry, producing high-dimensional "
        "sparse representations with low per-feature frequency counts — a regime that "
        "disadvantages all frequency-based classifiers. Two principled alternatives "
        "have been explored in the literature: character-level n-gram features and "
        "sub-word tokenisation."))
    body_para(doc, (
        "Character n-gram TF-IDF represents each document as a bag of overlapping "
        "character substrings of length n. For a morphologically rich language, character "
        "n-grams capture shared substrings across morphological variants, providing "
        "implicit morphological smoothing without requiring a formal morphological "
        "analyser. Bojanowski et al. (2017) demonstrated this principle for FastText "
        "embeddings across multiple morphologically rich languages; the present study "
        "demonstrates an analogous benefit in the TF-IDF feature space for Amharic. "
        "Byte Pair Encoding (BPE) and SentencePiece tokenisation (Kudo & Richardson, 2018) "
        "provide an alternative sub-word approach, learning a vocabulary of common "
        "character sequences from corpus statistics. BPE is used in the transformer "
        "experiments reported in Chapter 4."))

    section_heading(doc, "2.1.6 Socio-Technical Systems Theory", level=2)
    body_para(doc, (
        "Building on the socio-technical systems perspective (Trist & Bamforth, 1951; "
        "Emery, 1959), this study acknowledges that AI sentiment analysis tools cannot "
        "succeed in isolation from the social environments they serve. Effective deployment "
        "of Amharic sentiment tools for public policy requires alignment with institutional "
        "workflows, trust from government stakeholders, and consideration of cultural norms "
        "around the expression of political opinion in Ethiopian social contexts. "
        "Socio-technical theory guides the study's emphasis on practical deployability, "
        "interpretability, and the identification of future work that bridges the gap "
        "between model development and real-world application."))

    section_heading(doc, "2.1.7 AI Frameworks for Public Policy and Governance", level=2)
    body_para(doc, (
        "Public policy frameworks increasingly emphasise evidence-based decision-making "
        "and participatory governance. The integration of AI into these frameworks "
        "requires attention to transparency, accountability, and the ethical use of "
        "citizen data (Doshi-Velez & Kim, 2017). For Ethiopian government contexts, "
        "AI sentiment tools must be interpretable (policymakers must understand what "
        "the model predicts and why), computationally accessible (deployable without "
        "cloud GPU infrastructure), and linguistically appropriate (operating natively "
        "on Amharic text). The MNB + character n-gram model developed in this study "
        "satisfies all three criteria."))

    # ── 2.2 Empirical ────────────────────────────────────────────────────────
    section_heading(doc, "2.2 Empirical Literature Review")

    section_heading(doc, "2.2.1 Sentiment Analysis in High-Resource Languages", level=2)
    body_para(doc, (
        "Sentiment analysis has matured significantly in high-resource language settings, "
        "particularly English. Early work by Pang et al. (2002) established SVM and "
        "Naïve Bayes as effective baselines for document-level sentiment classification. "
        "Socher et al. (2013) introduced the Stanford Sentiment Treebank and demonstrated "
        "that recursive neural networks could capture compositional sentiment. The BERT "
        "revolution (Devlin et al., 2019) led to near-human performance on English sentiment "
        "benchmarks through fine-tuning. These advances, however, depend on millions of "
        "labelled examples and large-scale pretraining corpora that do not exist for Amharic."))
    body_para(doc, (
        "Applications of sentiment analysis for public policy monitoring in developed "
        "countries provide the applied motivation for the present study. During the COVID-19 "
        "pandemic, sentiment analysis systems tracked public compliance, vaccine hesitancy, "
        "and misinformation spread in real time, enabling rapid policy adjustments (Qorib "
        "et al., 2023). South Korea, the United States, and the United Kingdom have "
        "integrated social media sentiment monitoring into formal policy evaluation processes. "
        "These applications demonstrate the practical value of the technology that this "
        "study seeks to make accessible for Amharic and Ethiopia."))

    section_heading(doc, "2.2.2 Sentiment Analysis in African Languages", level=2)
    body_para(doc, (
        "The AfriSenti project (Muhammad et al., 2023) represents the most significant "
        "recent advance in African language sentiment analysis. AfriSenti released "
        "annotated sentiment datasets for 14 African languages — including Amharic, "
        "Hausa, Yoruba, Igbo, Swahili, and others — collected from Twitter and annotated "
        "by native speakers under a standardised three-class protocol. The SemEval-2023 "
        "Task 12 shared task, based on AfriSenti, attracted 44 teams and established the "
        "first multilingual African sentiment benchmark. Top-performing teams used "
        "fine-tuned Afro-XLM-R, achieving weighted F1 scores of 0.65–0.72 on the "
        "Amharic test set."))
    body_para(doc, (
        "Azime et al. (2023) described the Masakhane-AfriSenti submission, which used "
        "Afro-centric language model adapters for low-resource African languages in "
        "the SemEval-2023 task. Their approach demonstrated that African-language-specific "
        "adaptation of multilingual models provides measurable improvements over "
        "off-the-shelf XLM-RoBERTa. Alabi et al. (2022) introduced multilingual adaptive "
        "fine-tuning (MAFT) of pretrained models on African languages, establishing the "
        "approach that subsequent Afro-XLM-R systems have used."))

    section_heading(doc, "2.2.3 Amharic NLP and Sentiment Research", level=2)
    body_para(doc, (
        "Amharic NLP research has grown substantially since 2020. Yimam et al. (2020) "
        "published the first systematic exploration of Amharic sentiment analysis using "
        "social media texts, building annotation tools and classification models that "
        "established foundational preprocessing steps including Ethiopic character "
        "normalisation. Their work highlighted the challenge of orthographic inconsistency "
        "in informal Amharic writing — multiple equivalent representations for the same "
        "phoneme — that the preprocessing pipeline in the present study directly addresses."))
    body_para(doc, (
        "Alemayehu et al. (2023) trained a CNN-BiLSTM model on a corpus of Amharic "
        "political texts from social media, reporting 91.6% accuracy on binary sentiment "
        "classification. While this result appears high, it is not directly comparable "
        "to the present study: the corpus was restricted to political text (a stylistically "
        "homogeneous domain), the task was binary rather than three-class, and the metric "
        "was accuracy rather than macro-F1. Accuracy on imbalanced binary datasets "
        "substantially overestimates practical classifier performance."))
    body_para(doc, (
        "Alemayehu and Aseres (2022) reviewed Amharic sentiment analysis literature, "
        "identifying the scarcity of annotated corpora and computational infrastructure "
        "as the primary bottlenecks, and recommending the development of standardised "
        "benchmarks — a recommendation that the AfriSenti project subsequently fulfilled. "
        "Aynalem (2022) conducted rule-based sentiment analysis on Tigrinya television "
        "service reviews, providing a complementary perspective on Ethiopic-script "
        "sentiment analysis that highlights the limitations of lexicon-based approaches "
        "for under-resourced Semitic languages."))
    body_para(doc, (
        "Azime and Mohammed (2023) applied transfer learning approaches to Amharic "
        "sentiment analysis, demonstrating that multilingual pretrained models could "
        "be effectively adapted to Amharic when fine-tuning data was available. Their "
        "results confirmed the theoretical expectation that pretrained representations "
        "are the primary driver of transformer performance on small Amharic datasets."))

    section_heading(doc, "2.2.4 AfriSenti and Multilingual Benchmarks", level=2)
    body_para(doc, (
        "The AfriSenti benchmark dataset (Muhammad et al., 2023) provides the evaluation "
        "framework for this study. The Amharic subset contains approximately 8,950 "
        "annotated tweets collected from Twitter/X, with standardised train, development, "
        "and test splits. Tweets were annotated by native Amharic speakers using a "
        "three-class scheme (positive, negative, neutral) following detailed annotation "
        "guidelines developed specifically for African social media text. The corpus "
        "reflects the diversity of Ethiopian public discourse on social, political, "
        "and cultural topics."))
    body_para(doc, (
        "The SemEval-2023 Task 12 shared task evaluation (Muhammad et al., 2023) "
        "provided the first systematic comparison of submitted systems on the AfriSenti "
        "benchmark. For the Amharic subtask, the top systems achieved weighted F1 scores "
        "in the range 0.65–0.72, predominantly using fine-tuned XLM-RoBERTa. The "
        "baseline provided by the shared task organisers achieved approximately 0.60 "
        "weighted F1 using a Naïve Bayes classifier with word-level TF-IDF features. "
        "The present study improves on this baseline by 12+ F1 points through systematic "
        "feature engineering, without any architectural changes."))

    # ── Table 2.1 ────────────────────────────────────────────────────────────
    lit_headers = ["Author(s), Year", "Language", "Model", "Best Score", "Dataset", "Limitations"]
    lit_rows = [
        ["Yimam et al. (2020)", "Amharic", "NB, SVM, CNN", "F1≈0.60", "Custom social media", "Binary; small corpus"],
        ["Tessema & Yimam (2021)", "Amharic", "Bi-LSTM", "Acc=0.73", "Custom (binary)", "Binary; domain-specific"],
        ["Alemayehu & Aseres (2022)", "Amharic", "Review paper", "N/A", "Literature survey", "No experiments"],
        ["Alemayehu et al. (2023)", "Amharic", "CNN-BiLSTM", "Acc=0.916", "Political corpus", "Binary; single domain; accuracy only"],
        ["Ayele et al. (2023)", "Amharic", "XLM-RoBERTa", "F1=0.72", "AfriSenti", "GPU required; no classical baseline"],
        ["Azime et al. (2023)", "Multi-African", "Afro-XLM-R", "F1=0.65–0.72", "AfriSenti SemEval", "GPU required"],
        ["Muhammad et al. (2023)", "14 African langs", "Baseline NB", "F1≈0.60", "AfriSenti", "Baseline only; no deep ablation"],
        ["This study (2025)", "Amharic", "MNB + Char n-grams", "F1=0.7282", "AfriSenti + policy", "No pretrained transformers"],
    ]
    make_table(doc, lit_headers, lit_rows,
               "Table 2.1: Summary of Related Works on Amharic Sentiment Analysis")

    # ── 2.3 Research Gap ─────────────────────────────────────────────────────
    section_heading(doc, "2.3 Research Gap")
    body_para(doc, (
        "The review of existing literature reveals three principal gaps that this study "
        "addresses. First, no prior published study has conducted a systematic, head-to-head "
        "comparison of as many as ten model families — spanning classical ML, gradient-"
        "boosted ensembles, deep learning, and transformers — on the official AfriSenti "
        "Amharic benchmark using consistent preprocessing and evaluation protocols. The "
        "fragmentation of prior work across different datasets, metrics, and model subsets "
        "makes cross-study comparisons unreliable and prevents practitioners from making "
        "informed model selection decisions."))
    body_para(doc, (
        "Second, feature engineering for Amharic morphology has been systematically "
        "neglected. Prior studies default to word-level TF-IDF without evaluating "
        "whether character-level or sub-word features are better suited to Amharic's "
        "agglutinative morphology. The potential for character n-gram TF-IDF to "
        "dramatically improve performance — grounded in theoretical principles of "
        "morphological smoothing — had not been tested in the Amharic sentiment context. "
        "This gap motivated the 108-experiment feature engineering ablation that is the "
        "central experimental contribution of this thesis."))
    body_para(doc, (
        "Third, practical deployability under resource constraints has been underemphasised. "
        "High-performing studies require GPU infrastructure and pretrained transformer "
        "weights not available in most Ethiopian institutional settings. The present study "
        "explicitly targets CPU-only deployment, producing a model — MNB with character "
        "n-gram TF-IDF — that is accessible to practitioners without specialised "
        "computational infrastructure."))

    # ── 2.4 Conceptual Framework ─────────────────────────────────────────────
    section_heading(doc, "2.4 Conceptual Framework")
    body_para(doc, (
        "The conceptual framework for this study (Figure 3.1) illustrates the sequential "
        "pipeline through which raw Amharic social media data is transformed into actionable "
        "sentiment insights for public policy. The framework is grounded in the socio-technical "
        "systems perspective: it emphasises not only the technical components of the "
        "pipeline but also the institutional and social contexts that determine whether "
        "the pipeline produces useful outputs for policymakers."))
    body_para(doc, (
        "The framework comprises five components in strict sequential dependency. Data "
        "collection produces a raw annotated corpus of Amharic tweets from AfriSenti and "
        "supplementary policy-domain sources. Preprocessing transforms the raw text into "
        "a clean, normalised, and consistently encoded representation. Feature extraction "
        "converts the preprocessed text into numerical feature vectors suitable for "
        "machine learning. Model training and evaluation identifies the best-performing "
        "classifier through systematic experimentation. Finally, sentiment trend analysis "
        "applies the selected model to new text to generate policy-relevant insights "
        "for government stakeholders."))
    body_para(doc, (
        "The conceptual framework is operationalised in Chapter Three's methodology "
        "section, which describes each component in detail. Figure 3.1 (presented in "
        "Chapter Three) provides a visual representation of the pipeline architecture."))
    page_break(doc)

print("Chapter 2 function defined.")
