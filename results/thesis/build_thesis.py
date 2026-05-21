"""
Thesis builder — Aklilu Gebeyehu MSc Thesis
Developing Deep Learning-Based Sentiment Analysis of Amharic Social Media
for Public Policy Enhancement in Ethiopia
"""

import os, json, csv
from datetime import datetime
from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

BASE = "/home/user/Aklil-amharic-sentiment-analysis"
FIG  = os.path.join(BASE, "results/final_figures")
OUT  = os.path.join(BASE, "results/thesis")

# ── helpers ──────────────────────────────────────────────────────────────────

def set_page_size(doc):
    section = doc.sections[0]
    section.page_height = Cm(29.7)
    section.page_width  = Cm(21.0)
    section.top_margin    = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin   = Inches(1.5)
    section.right_margin  = Inches(1)

def set_para_format(para, first_indent=True, space_before=0, space_after=0,
                    line_spacing=1.5, alignment=WD_ALIGN_PARAGRAPH.JUSTIFY):
    pf = para.paragraph_format
    pf.alignment = alignment
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = line_spacing
    pf.space_before = Pt(space_before)
    pf.space_after  = Pt(space_after)
    if first_indent:
        pf.first_line_indent = Inches(0.5)
    else:
        pf.first_line_indent = Pt(0)

def add_run(para, text, bold=False, italic=False, size=12, font="Times New Roman"):
    run = para.add_run(text)
    run.bold   = bold
    run.italic = italic
    run.font.name = font
    run.font.size = Pt(size)
    return run

def body_para(doc, text, first_indent=True, bold=False, size=12):
    p = doc.add_paragraph()
    set_para_format(p, first_indent=first_indent)
    add_run(p, text, bold=bold, size=size)
    return p

def chapter_title(doc, text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf.space_before = Pt(24)
    pf.space_after  = Pt(12)
    pf.first_line_indent = Pt(0)
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = 1.5
    add_run(p, text.upper(), bold=True, size=14)
    return p

def section_heading(doc, text, level=1):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.first_line_indent = Pt(0)
    pf.space_before = Pt(12)
    pf.space_after  = Pt(6)
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = 1.5
    sz = 13 if level == 1 else 12
    add_run(p, text, bold=True, size=sz)
    return p

def add_figure(doc, fig_path, caption, width=5.5):
    if os.path.exists(fig_path):
        p = doc.add_paragraph()
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Pt(0)
        run = p.add_run()
        run.add_picture(fig_path, width=Inches(width))
    else:
        p = doc.add_paragraph(f"[Insert {caption}]")
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.first_line_indent = Pt(0)
    cap.paragraph_format.space_after = Pt(12)
    add_run(cap, caption, size=11, italic=True)

def shade_row(row, hex_color="D9D9D9"):
    for cell in row.cells:
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), hex_color)
        tcPr.append(shd)

def make_table(doc, headers, rows, caption, bold_col0=False):
    cap = doc.add_paragraph()
    cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    cap.paragraph_format.first_line_indent = Pt(0)
    cap.paragraph_format.space_before = Pt(12)
    cap.paragraph_format.space_after  = Pt(3)
    add_run(cap, caption, bold=True, size=11)

    ncols = len(headers)
    tbl = doc.add_table(rows=1+len(rows), cols=ncols)
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER

    hdr_row = tbl.rows[0]
    shade_row(hdr_row)
    for i, h in enumerate(headers):
        cell = hdr_row.cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_run(p, h, bold=True, size=11)

    for ri, row_data in enumerate(rows):
        trow = tbl.rows[ri+1]
        for ci, val in enumerate(row_data):
            cell = trow.cells[ci]
            cell.text = ""
            p = cell.paragraphs[0]
            align = WD_ALIGN_PARAGRAPH.LEFT if ci == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.alignment = align
            add_run(p, str(val), bold=(bold_col0 and ci == 0), size=11)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)
    return tbl

def page_break(doc):
    doc.add_page_break()

# ── FRONT MATTER ─────────────────────────────────────────────────────────────

def build_title_page(doc):
    for _ in range(4):
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.space_after = Pt(0)

    def centred(text, bold=False, size=12, space_after=6):
        p = doc.add_paragraph()
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.space_after = Pt(space_after)
        add_run(p, text, bold=bold, size=size)

    centred("TECHNICAL AND VOCATIONAL TRAINING INSTITUTE", bold=True, size=13)
    centred("SCHOOL OF GRADUATE STUDIES", bold=True, size=13)
    centred("FACULTY OF ELECTRICAL ELECTRONICS AND", bold=True, size=13)
    centred("INFORMATION COMMUNICATION TECHNOLOGY", bold=True, size=13)
    centred("DEPARTMENT OF INFORMATION COMMUNICATION TECHNOLOGY (ICT)", bold=True, size=13, space_after=36)
    centred("Developing Deep Learning-Based Sentiment Analysis of Amharic", bold=True, size=14, space_after=0)
    centred("Social Media for Public Policy Enhancement in Ethiopia", bold=True, size=14, space_after=36)
    centred("A Thesis Submitted in Partial Fulfillment of the Requirements for the", size=12, space_after=0)
    centred("Degree of Master of Science in Information Communication Technology", size=12, space_after=36)
    centred("BY:", bold=True, size=12, space_after=6)
    centred("Aklilu Gebeyehu", bold=True, size=12, space_after=24)
    centred("Advisor: Dr. Martha Yifiru", size=12, space_after=48)
    centred("Addis Ababa, Ethiopia", size=12, space_after=6)
    centred("2025", size=12)
    page_break(doc)

def build_declaration(doc):
    chapter_title(doc, "DECLARATION")
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_after = Pt(12)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    p.paragraph_format.line_spacing = 1.5
    add_run(p, ('I declare that this thesis entitled "Developing Deep Learning-Based Sentiment Analysis '
                'of Amharic Social Media for Public Policy Enhancement in Ethiopia" is my own work and '
                'that it has not been presented in any other institution for any degree or award. All '
                'sources of materials used have been duly acknowledged.'), size=12)
    for line in ["Name of Student: ______________________",
                 "Signature: ___________________________",
                 "Date of Submission: ___________________"]:
        p2 = doc.add_paragraph()
        p2.paragraph_format.first_line_indent = Pt(0)
        p2.paragraph_format.space_after = Pt(12)
        add_run(p2, line, size=12)
    page_break(doc)

def build_certification(doc):
    chapter_title(doc, "CERTIFICATION")
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_after = Pt(12)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    p.paragraph_format.line_spacing = 1.5
    add_run(p, ('This is to certify that this thesis entitled "Developing Deep Learning-Based Sentiment '
                'Analysis of Amharic Social Media for Public Policy Enhancement in Ethiopia" has been '
                'submitted for examination with my approval as a university advisor.'), size=12)
    for line in [
        "Advisor: Dr. Martha Yifiru",
        "Signature: ___________________________",
        "Date: ___________________________",
        "",
        "Internal Examiner: ______________________",
        "Signature: ___________________________",
        "Date: ___________________________",
        "",
        "External Examiner: ______________________",
        "Signature: ___________________________",
        "Date: ___________________________",
    ]:
        p2 = doc.add_paragraph()
        p2.paragraph_format.first_line_indent = Pt(0)
        p2.paragraph_format.space_after = Pt(10)
        add_run(p2, line, size=12)
    page_break(doc)

def build_acknowledgment(doc):
    chapter_title(doc, "ACKNOWLEDGMENT")
    body_para(doc, ("The researcher wishes to express sincere gratitude to Dr. Martha Yifiru for her "
                    "unwavering guidance, critical feedback, and encouragement throughout the course of "
                    "this research. Her insights into computational linguistics and her commitment to "
                    "rigorous scholarship shaped every phase of this work."), first_indent=True)
    body_para(doc, ("Appreciation is also extended to the faculty of Electrical Electronics and Information "
                    "Communication Technology at the Technical and Vocational Training Institute for "
                    "providing an enabling academic environment. The researcher is grateful to colleagues "
                    "and peers who contributed valuable discussions and constructive criticism."), first_indent=True)
    body_para(doc, ("The AfriSenti research team, whose publicly released Amharic benchmark dataset made "
                    "this study possible, deserves special acknowledgment. Finally, heartfelt thanks are "
                    "due to family and friends for their patience and moral support throughout the "
                    "demanding process of graduate research."), first_indent=True)
    page_break(doc)

def build_abstract(doc):
    chapter_title(doc, "ABSTRACT")
    body_para(doc, ("Amharic, the official language of Ethiopia and a mother tongue of over 57 million "
                    "speakers, remains severely under-resourced in computational natural language processing. "
                    "As Ethiopian citizens increasingly use social media platforms such as Twitter/X, "
                    "Facebook, and Telegram to express opinions on governance and public policy, the need "
                    "for automated Amharic sentiment analysis tools has become urgent. Existing studies "
                    "have evaluated isolated model families on non-standardised datasets, making "
                    "cross-study comparisons unreliable."), first_indent=True)
    body_para(doc, ("This study addresses that gap by conducting the first systematic, head-to-head "
                    "benchmark of ten model families on the official AfriSenti Amharic dataset: Multinomial "
                    "Naïve Bayes (MNB), Logistic Regression, Support Vector Machine, K-Nearest Neighbour, "
                    "XGBoost, LightGBM, Bidirectional LSTM, a custom transformer, an MNB–Transformer "
                    "ensemble, and a stacking classifier. A total of 108 experiments were conducted across "
                    "four systematic phases exploring word-level and character-level TF-IDF feature "
                    "representations, n-gram ranges, vocabulary sizes, and smoothing parameters."), first_indent=True)
    body_para(doc, ("The key finding is that character-level TF-IDF n-gram features (range 2–5, "
                    "max_features=80,000) dramatically outperform word-level features for Amharic, yielding "
                    "an absolute improvement of +11.66 macro-F1 points. The best model—MNB with character "
                    "n-gram TF-IDF and Laplace smoothing α=0.2—achieved macro-F1=0.7282 on the held-out "
                    "test set (n=1,646), trained in 1.9 seconds on a standard CPU without GPU resources. "
                    "This result is competitive with the top teams in the AfriSenti SemEval-2023 shared "
                    "task, who used GPU-trained fine-tuned XLM-RoBERTa. The transformer trained from "
                    "random initialisations achieved macro-F1=0.3648, confirming that pretrained "
                    "representations—not architectural complexity—drive transformer performance on small "
                    "Amharic datasets."), first_indent=True)
    body_para(doc, ("The study contributes a reproducible open-source pipeline, a comprehensive ablation "
                    "of feature engineering choices for morphologically rich Amharic text, and empirical "
                    "evidence that carefully designed classical models can match GPU-trained deep learning "
                    "systems in low-resource conditions. Practical recommendations are provided for "
                    "NLP researchers, system developers, and Ethiopian policymakers seeking to deploy "
                    "Amharic sentiment analysis tools for public policy monitoring."), first_indent=True)
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_before = Pt(12)
    add_run(p, "Keywords: ", bold=True, size=12)
    add_run(p, ("Amharic sentiment analysis, character n-grams, TF-IDF, Multinomial Naïve Bayes, "
                "deep learning, low-resource NLP, AfriSenti, public policy, Ethiopia"), size=12)
    page_break(doc)

def build_abbreviations(doc):
    chapter_title(doc, "LIST OF ABBREVIATIONS")
    abbrevs = [
        ("AfriSenti", "African Sentiment Analysis Benchmark"),
        ("BPE",       "Byte Pair Encoding"),
        ("Bi-LSTM",   "Bidirectional Long Short-Term Memory"),
        ("CNN",       "Convolutional Neural Network"),
        ("DL",        "Deep Learning"),
        ("EDA",       "Exploratory Data Analysis"),
        ("F1",        "F1-Score (harmonic mean of precision and recall)"),
        ("FN",        "False Negative"),
        ("FP",        "False Positive"),
        ("KNN",       "K-Nearest Neighbor"),
        ("LGBM",      "Light Gradient Boosting Machine"),
        ("LR",        "Logistic Regression"),
        ("ML",        "Machine Learning"),
        ("MNB",       "Multinomial Naive Bayes"),
        ("NLP",       "Natural Language Processing"),
        ("SA",        "Sentiment Analysis"),
        ("SVM",       "Support Vector Machine"),
        ("TF-IDF",    "Term Frequency-Inverse Document Frequency"),
        ("TN",        "True Negative"),
        ("TP",        "True Positive"),
        ("XGB",       "Extreme Gradient Boosting"),
        ("XLM-R",     "Cross-lingual Language Model - RoBERTa"),
    ]
    for abbr, meaning in abbrevs:
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
        p.paragraph_format.line_spacing = 1.5
        add_run(p, f"{abbr:<12}", bold=True, size=12)
        add_run(p, f"  {meaning}", size=12)
    page_break(doc)

def build_toc(doc):
    chapter_title(doc, "TABLE OF CONTENTS")
    entries = [
        ("Declaration", "i"),
        ("Certification", "ii"),
        ("Acknowledgment", "iii"),
        ("Abstract", "iv"),
        ("List of Abbreviations", "v"),
        ("Table of Contents", "vi"),
        ("List of Tables", "vii"),
        ("List of Figures", "viii"),
        ("CHAPTER ONE: INTRODUCTION", "1"),
        ("  1.1 Background of the Study", "1"),
        ("  1.2 Motivation of the Study", "3"),
        ("  1.3 Statement of the Problem", "4"),
        ("  1.4 Objectives of the Study", "6"),
        ("  1.5 Research Questions", "7"),
        ("  1.6 Scope of the Study", "7"),
        ("  1.7 Significance of the Study", "8"),
        ("  1.8 Limitations of the Study", "9"),
        ("  1.9 Organization of the Thesis", "10"),
        ("CHAPTER TWO: REVIEW OF RELATED LITERATURE", "11"),
        ("  2.1 Theoretical Literature Review", "11"),
        ("  2.2 Empirical Literature Review", "18"),
        ("  2.3 Research Gap", "23"),
        ("  2.4 Conceptual Framework", "24"),
        ("CHAPTER THREE: RESEARCH METHODOLOGY", "26"),
        ("  3.1 Research Design", "26"),
        ("  3.2 Dataset Description", "27"),
        ("  3.3 Data Preprocessing", "28"),
        ("  3.4 Feature Extraction", "30"),
        ("  3.5 Model Architectures", "32"),
        ("  3.6 Evaluation Metrics", "37"),
        ("  3.7 Experimental Setup", "39"),
        ("  3.8 Ethical Considerations", "39"),
        ("CHAPTER FOUR: RESULTS AND DISCUSSION", "41"),
        ("  4.1 Dataset Characteristics", "41"),
        ("  4.2 Preprocessing Results", "43"),
        ("  4.3 Feature Engineering Analysis", "45"),
        ("  4.4 Machine Learning Results", "48"),
        ("  4.5 Ensemble Methods", "51"),
        ("  4.6 Deep Learning Results", "52"),
        ("  4.7 Comparative Analysis", "57"),
        ("  4.8 Ablation Study", "59"),
        ("  4.9 Comparison with Prior Work", "61"),
        ("  4.10 Discussion of Findings", "62"),
        ("  4.11 Limitations", "64"),
        ("CHAPTER FIVE: CONCLUSION AND FUTURE WORK", "67"),
        ("  5.1 Summary", "67"),
        ("  5.2 Conclusions: Research Questions Answered", "69"),
        ("  5.3 Contributions", "70"),
        ("  5.4 Recommendations", "73"),
        ("  5.5 Future Work", "75"),
        ("  5.6 Closing Remarks", "78"),
        ("REFERENCES", "79"),
        ("APPENDICES", "85"),
    ]
    for entry, page in entries:
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
        p.paragraph_format.line_spacing = 1.5
        bold = not entry.startswith("  ")
        tab = doc.add_paragraph()
        tab.clear()
        p2 = doc.add_paragraph()  # won't use this
        # Simpler: just add the text directly
        p.clear()
        run1 = p.add_run(entry)
        run1.bold = bold
        run1.font.name = "Times New Roman"
        run1.font.size = Pt(12)
        run2 = p.add_run(f"\t{page}")
        run2.font.name = "Times New Roman"
        run2.font.size = Pt(12)
    page_break(doc)

def build_list_of_tables(doc):
    chapter_title(doc, "LIST OF TABLES")
    tables = [
        ("Table 3.1", "Dataset Statistics — AfriSenti Amharic Corpus"),
        ("Table 3.2", "Hyperparameter Configurations for All Models"),
        ("Table 4.1", "Dataset Split Summary"),
        ("Table 4.2", "Class Distribution in the Full Corpus"),
        ("Table 4.3", "Tweet Length Statistics"),
        ("Table 4.4", "Character N-grams vs. Word N-grams: MNB Performance"),
        ("Table 4.5", "N-gram Range Ablation: MNB + Char TF-IDF"),
        ("Table 4.6", "Max Features Ablation: MNB + Char TF-IDF"),
        ("Table 4.7", "Machine Learning Model Comparison"),
        ("Table 4.8", "MNB + Char N-grams Per-Class Performance"),
        ("Table 4.9", "Gradient Boosting Results"),
        ("Table 4.10", "Bi-LSTM Performance"),
        ("Table 4.11", "Bi-LSTM Per-Class Performance"),
        ("Table 4.12", "Transformer (Random Init) Performance"),
        ("Table 4.13", "Transformer (Random Init) Per-Class Performance"),
        ("Table 4.14", "MNB–Transformer Ensemble α-Sweep"),
        ("Table 4.15", "Full Model Leaderboard"),
        ("Table 4.16", "Experimental Programme Summary"),
        ("Table 4.17", "Comparison with Published Amharic Sentiment Analysis Results"),
    ]
    for tbl_id, tbl_name in tables:
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
        p.paragraph_format.line_spacing = 1.5
        add_run(p, f"{tbl_id}:  ", bold=True, size=12)
        add_run(p, tbl_name, size=12)
        add_run(p, "\t...", size=12)
    page_break(doc)

def build_list_of_figures(doc):
    chapter_title(doc, "LIST OF FIGURES")
    figures = [
        ("Figure 3.1", "Research Methodology Pipeline"),
        ("Figure 4.1", "Label Distribution Across Sentiment Classes"),
        ("Figure 4.2", "Tweet Length Distribution by Sentiment Class"),
        ("Figure 4.3", "Character N-grams vs. Word N-grams Performance Comparison"),
        ("Figure 4.4", "All Models Performance Comparison"),
        ("Figure 4.5", "Best Model Confusion Matrix (MNB + Char N-grams)"),
        ("Figure 4.6", "Per-Class F1 Heatmap Across All Models"),
        ("Figure 4.7", "Experiment Progression Across Four Phases"),
        ("Figure 4.8", "Bi-LSTM Training and Validation Curves"),
        ("Figure 4.9", "Learning Curve for MNB and Comparison Models"),
    ]
    for fig_id, fig_name in figures:
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
        p.paragraph_format.line_spacing = 1.5
        add_run(p, f"{fig_id}:  ", bold=True, size=12)
        add_run(p, fig_name, size=12)
        add_run(p, "\t...", size=12)
    page_break(doc)

print("Front matter functions defined.")
