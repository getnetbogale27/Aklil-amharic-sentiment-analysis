"""Chapter 1, 2, 3 content functions."""

from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from build_thesis import (body_para, chapter_title, section_heading,
                           add_run, make_table, page_break, add_figure,
                           set_para_format, FIG)

def build_chapter1(doc):
    chapter_title(doc, "CHAPTER ONE")
    chapter_title(doc, "INTRODUCTION")

    section_heading(doc, "1.1 Background of the Study")
    body_para(doc, (
        "Sentiment analysis, also referred to as opinion mining, is a subfield of Natural Language Processing "
        "(NLP) that focuses on the computational identification and classification of opinions, emotions, and "
        "attitudes expressed in text data (Liu, 2022). By automatically determining whether a piece of text "
        "conveys positive, negative, or neutral sentiment, sentiment analysis systems enable organisations, "
        "governments, and researchers to process large volumes of user-generated content at a scale that "
        "manual analysis cannot achieve. Applications span commercial product review analysis, healthcare "
        "monitoring, political discourse tracking, and — most relevantly for this study — public policy "
        "evaluation (Pang & Lee, 2008)."))
    body_para(doc, (
        "The rapid proliferation of social media platforms has fundamentally transformed the landscape of "
        "public discourse in Ethiopia and across the African continent. As of 2023, Ethiopia's social media "
        "user base exceeded seven million on Facebook and ten million on Telegram, with Twitter/X serving "
        "as the primary platform for political commentary and policy debate among urban, educated, and "
        "diaspora communities (Muhammad et al., 2023; Tonja et al., 2023). These platforms provide "
        "unprecedented access to real-time expressions of public opinion on government policies, economic "
        "conditions, social issues, and governance outcomes — a digital commons that, if systematically "
        "analysed, could provide policymakers with evidence-based insights into citizen concerns."))
    body_para(doc, (
        "Amharic, the official working language of the Federal Democratic Republic of Ethiopia, is spoken "
        "by more than 57 million people as a first or second language, making it one of the most widely "
        "spoken languages in Africa (Beshada & Balcha, 2023). Despite this demographic significance, "
        "Amharic remains severely under-resourced in the field of NLP. The language presents several "
        "computational challenges: it is written in Ethiopic (Ge'ez) script, a unique abugida system "
        "not shared with any other language; it possesses highly agglutinative morphology in which a "
        "single root generates dozens of surface forms through affixation of tense, person, number, "
        "gender, and case; and social-media usage introduces additional complexity through "
        "code-switching, informal orthography, and dialectal variation (Aynalem, 2022; Yimam et al., 2020)."))
    body_para(doc, (
        "Prior to the release of the AfriSenti benchmark dataset (Muhammad et al., 2023), annotated "
        "Amharic sentiment data was scarce, domain-specific, and methodologically heterogeneous. "
        "Studies that did exist often focused on narrow political corpora, used binary rather than "
        "three-class sentiment classification, or reported results that could not be compared across "
        "studies due to differing evaluation protocols. The absence of a standardised benchmark "
        "meant that each new study effectively established its own baseline, making cumulative "
        "scientific progress difficult to measure."))
    body_para(doc, (
        "This thesis responds to that gap by conducting a systematic, reproducible benchmark of ten "
        "model families — spanning classical machine learning, gradient-boosted ensembles, deep "
        "learning, and transformer architectures — on the official AfriSenti Amharic benchmark, "
        "supplemented by a policy-domain tweet corpus. The study is motivated by the practical need "
        "for deployable Amharic sentiment tools that can operate within the computational constraints "
        "common in Ethiopian institutional settings, and by the scientific need for a rigorous, "
        "comparative foundation for future Amharic NLP research."))

    section_heading(doc, "1.2 Motivation of the Study")
    body_para(doc, (
        "The primary practical motivation for this study is the digital transformation of Ethiopian "
        "public discourse. Government ministries, civil-society organisations, and research institutions "
        "increasingly recognise that social media platforms host a rich and continuously updated stream "
        "of citizen opinion. However, manually monitoring, categorising, and synthesising the volume of "
        "Amharic social media content generated daily — estimated at hundreds of thousands of posts "
        "across platforms — is infeasible without automated tools."))
    body_para(doc, (
        "Current sentiment analysis tools are developed almost exclusively for high-resource languages "
        "such as English, Chinese, and Arabic, which benefit from large annotated corpora, mature "
        "pretrained language models, and active commercial investment in NLP infrastructure. Applying "
        "these tools directly to Amharic is not viable due to the script barrier, the morphological "
        "incompatibility with tokenisation strategies designed for Latin-script languages, and the "
        "absence of Amharic in most commercial sentiment lexicons."))
    body_para(doc, (
        "A secondary motivation is scientific. The release of the AfriSenti benchmark in 2023 "
        "created, for the first time, a standardised evaluation framework for Amharic sentiment "
        "analysis. Yet the accompanying baseline analysis for the Amharic subset was limited, and "
        "subsequent studies have used different evaluation setups, making results incomparable. "
        "A systematic, head-to-head comparison of model families on the same dataset, using "
        "consistent preprocessing and evaluation metrics, is a scientific contribution with lasting "
        "utility for the field."))

    section_heading(doc, "1.3 Statement of the Problem")
    body_para(doc, (
        "Despite the global success of sentiment analysis in understanding public opinion across "
        "languages such as English and Arabic, its application to Amharic remains severely limited "
        "(Alemayehu & Aseres, 2022). This limitation arises from three compounding factors: the "
        "scarcity of annotated Amharic text corpora; the morphological and orthographic complexity "
        "of Amharic that renders standard NLP preprocessing pipelines inadequate; and the absence "
        "of systematic comparative studies that would allow practitioners to select the most "
        "appropriate model for a given resource constraint."))
    body_para(doc, (
        "Prior work on Amharic sentiment analysis has been characterised by methodological "
        "fragmentation. Tessema and Yimam (2021) focused on traditional ML classifiers; "
        "Alemayehu et al. (2023) evaluated CNN-BiLSTM architectures on a single-domain political "
        "corpus with binary classification; Ayele et al. (2023) applied fine-tuned XLM-RoBERTa "
        "to the AfriSenti benchmark but did not compare it against classical baselines. No "
        "published study has placed all three model families — classical ML, deep learning, and "
        "transformers — in a controlled, head-to-head comparison using the same data, "
        "preprocessing pipeline, and evaluation metric."))
    body_para(doc, (
        "Feature engineering for Amharic morphology has received insufficient attention. "
        "Most published studies default to word-level TF-IDF representations without "
        "systematically evaluating whether character-level or sub-word features are better "
        "suited to Amharic's agglutinative morphology. The hypothesis that character n-gram "
        "features may dramatically improve performance — grounded in the well-established "
        "finding that such features benefit morphologically rich languages (Bojanowski et al., "
        "2017; Mikolov et al., 2018) — had not been tested in the Amharic sentiment context "
        "prior to this work."))
    body_para(doc, (
        "Finally, practical deployability has been underemphasised in prior research. Studies "
        "that achieve high accuracy with GPU-trained transformer models provide limited guidance "
        "for practitioners working in Ethiopian government or civil-society contexts where GPU "
        "infrastructure is unavailable. This study explicitly examines the trade-off between "
        "model performance and computational cost, providing evidence-based guidance for "
        "resource-constrained deployment."))

    section_heading(doc, "1.4 Objectives of the Study")
    section_heading(doc, "1.4.1 General Objective", level=2)
    body_para(doc, (
        "To develop a deep learning-based sentiment analysis framework tailored to Amharic social "
        "media content for public policy enhancement in Ethiopia."), first_indent=False)

    section_heading(doc, "1.4.2 Specific Objectives", level=2)
    specifics = [
        "To investigate sentiment expressions and linguistic patterns in Amharic social media content related to public policy.",
        "To collect, preprocess, and prepare a representative dataset of Amharic social media posts for sentiment analysis.",
        "To design, implement, and evaluate machine learning and deep learning models for accurate Amharic sentiment classification.",
        "To compare model performance systematically across ML, DL, and transformer approaches using standardised evaluation metrics.",
    ]
    for i, obj in enumerate(specifics, 1):
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.left_indent = Inches(0.5)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
        p.paragraph_format.line_spacing = 1.5
        add_run(p, f"{i}. {obj}", size=12)

    section_heading(doc, "1.5 Research Questions")
    rqs = [
        "What are the prevalent sentiment expressions and linguistic patterns in Amharic social media content related to public policy?",
        "What methods can effectively collect, preprocess, and prepare Amharic social media data for sentiment analysis?",
        "How accurately can machine learning and deep learning models classify sentiment in Amharic social media content?",
    ]
    for i, rq in enumerate(rqs, 1):
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.left_indent = Inches(0.5)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
        p.paragraph_format.line_spacing = 1.5
        add_run(p, f"RQ{i}: {rq}", size=12)

    section_heading(doc, "1.6 Scope of the Study")
    body_para(doc, (
        "This study is scoped to the analysis of sentiment in Amharic-language social media "
        "content, specifically tweets from the AfriSenti benchmark and supplementary "
        "policy-domain tweets collected from Twitter/X. The classification task is three-class "
        "(positive, negative, neutral) sentiment classification. Other Ethiopian languages "
        "(Oromo, Tigrinya, Somali) and other sentiment-related tasks (aspect-level sentiment, "
        "emotion recognition, hate speech detection) are outside the scope of this study."))
    body_para(doc, (
        "The study evaluates ten model families under CPU-only computational constraints, "
        "reflecting the institutional context of Ethiopian universities and government agencies. "
        "Evaluation of pretrained multilingual transformers (XLM-RoBERTa, Afro-XLM-R) with "
        "GPU access is outside the current scope but is identified as priority future work. "
        "The study does not include live deployment of a policy-monitoring dashboard, "
        "which is also identified as future work."))

    section_heading(doc, "1.7 Significance of the Study")
    body_para(doc, (
        "This study makes a significant academic contribution as the first systematic benchmark "
        "of ten model families on the official AfriSenti Amharic dataset using a consistent "
        "preprocessing pipeline and evaluation protocol. The 108 systematic experiments provide "
        "the most comprehensive publicly available ablation study for Amharic sentiment feature "
        "engineering, offering a reproducible reference point for future research."))
    body_para(doc, (
        "The practical significance is equally important. The best-performing model — MNB with "
        "character n-gram TF-IDF — achieves macro-F1=0.7282 with training time of 1.9 seconds "
        "on a standard CPU, with no GPU requirement. This profile makes it immediately deployable "
        "in Ethiopian government, civil-society, and academic contexts where GPU infrastructure "
        "is unavailable. The model's trained artefacts are publicly available in the project "
        "repository, enabling immediate adoption by practitioners."))
    body_para(doc, (
        "Methodologically, the study's discovery that character n-gram features yield +11.66 "
        "macro-F1 points over word-level features has direct implications for any NLP project "
        "targeting Amharic or related morphologically rich Afroasiatic languages. The finding "
        "generalises the character n-gram advantage previously documented for European "
        "morphologically rich languages (Polish, Finnish, Turkish) to Ethiopic-script languages, "
        "extending the evidence base for this important NLP principle."))

    section_heading(doc, "1.8 Limitations of the Study")
    body_para(doc, (
        "The study is subject to several limitations that future work should address. First, "
        "the combined corpus of 10,972 usable samples is small by modern NLP standards, "
        "limiting the effectiveness of deep learning models trained from random initialisations. "
        "Second, access to pretrained transformer weights was unavailable in the execution "
        "environment, preventing evaluation of the strongest baseline in the literature "
        "(fine-tuned XLM-RoBERTa). Third, the AfriSenti annotation protocol provides no "
        "inter-annotator agreement statistics for the Amharic subset, introducing unknown "
        "label noise. Fourth, the evaluation is restricted to the AfriSenti train/dev/test "
        "domain and may not generalise to other Amharic text domains."))

    section_heading(doc, "1.9 Organization of the Thesis")
    body_para(doc, (
        "This thesis is organised into five chapters. Chapter One provides the background, "
        "motivation, problem statement, objectives, research questions, scope, and significance "
        "of the study. Chapter Two reviews the theoretical and empirical literature on sentiment "
        "analysis, machine learning, deep learning, transformer models, and Amharic NLP. "
        "Chapter Three describes the research methodology, including the dataset, preprocessing "
        "pipeline, feature extraction approaches, model architectures, evaluation metrics, and "
        "experimental setup. Chapter Four presents the results and discussion, covering "
        "dataset characteristics, feature engineering analysis, model comparisons, ablation "
        "studies, and comparison with prior work. Chapter Five concludes the thesis with "
        "a summary of findings, answers to the research questions, contributions, recommendations, "
        "and directions for future work."))
    page_break(doc)

print("Chapter 1 function defined.")
