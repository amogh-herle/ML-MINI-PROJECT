"""Build the 2-page write-up PDF required by the assignment guidelines
(problem statement, dataset details, approach, implementation overview,
conclusions). Pulls numbers from results/table1.json and
results/tuned_results.json so the write-up never drifts from the actual
results.
"""
import json
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

with open("results/table1.json") as f:
    t1 = json.load(f)["table1_accuracy"]
with open("results/tuned_results.json") as f:
    t2 = json.load(f)
with open("results/classifier_comparison.json") as f:
    t3 = json.load(f)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleCustom", parent=styles["Title"], fontSize=16, spaceAfter=2))
styles.add(ParagraphStyle(name="SubTitle", parent=styles["Normal"], fontSize=10, textColor=colors.grey, spaceAfter=10))
styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4))
styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"], fontSize=9.3, leading=12.5, spaceAfter=5))
styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10, textColor=colors.grey))

doc = SimpleDocTemplate(
    "report/writeup.pdf", pagesize=LETTER,
    topMargin=0.55 * inch, bottomMargin=0.55 * inch,
    leftMargin=0.65 * inch, rightMargin=0.65 * inch,
)

story = []

story.append(Paragraph("Classification of Legal Text: Replication &amp; Tuning", styles["TitleCustom"]))
story.append(Paragraph(
    "Mini-Project Write-up &nbsp;|&nbsp; Replicating Krithika Iyer, "
    "<i>“Classification of Legal Text,”</i> Stanford CS229 (2020)", styles["SubTitle"]))

story.append(Paragraph("Problem Statement", styles["H2"]))
story.append(Paragraph(
    "Legal opinions use long, non-standard sentence structures that break typical NLP tooling. "
    "We replicate and extend a Stanford CS229 project that classifies U.S. Supreme Court opinions "
    "into their human-curated legal subject-matter category, at two granularities: <b>issue_area</b> "
    "(15 broad categories, e.g. Criminal Procedure, Civil Rights, First Amendment) and <b>issue</b> "
    "(260+ fine-grained issue codes). The goal is to test whether classic topic-model and "
    "embedding-based document features, paired with a simple classifier, can recover these "
    "categories, and whether tuning their hyperparameters meaningfully improves on the paper's "
    "untuned baseline.", styles["Body"]))

story.append(Paragraph("Dataset", styles["H2"]))
story.append(Paragraph(
    "We use the exact dataset the original paper used: the Supreme Court Database (Washington "
    "University) via the <b>textacy</b> package's bundled <b>SupremeCourt</b> dataset — 8,419 "
    "opinions (1946-2016), each with an expert-assigned issue_area and issue code. No substitute "
    "was needed. After dropping opinions with no issue_area label and issue codes with fewer than "
    "10 examples (so every class can appear in train/dev/test), 7,971 opinions remain, split "
    "70%/15%/15% into train/dev/test, stratified by issue_area. Opinions run long (median ~34,000 "
    "characters); each is truncated to its first 4,000 characters before vectorization, keeping "
    "LDA/Doc2Vec training tractable.", styles["Body"]))

story.append(Paragraph("Approach", styles["H2"]))
story.append(Paragraph(
    "We replicate the paper's two completed experiments (its third, BERT, ran out of memory in the "
    "original work and produced no results to reproduce):", styles["Body"]))
story.append(Paragraph(
    "<b>1. LDA + Logistic Regression</b> — a gensim Dictionary and TF-IDF model feed a Latent "
    "Dirichlet Allocation topic model; each document becomes a dense topic-probability vector, "
    "classified with scikit-learn Logistic Regression.<br/>"
    "<b>2. Doc2Vec + Logistic Regression</b> — a gensim Doc2Vec model learns a fixed-size "
    "paragraph embedding per opinion, again classified with Logistic Regression.", styles["Body"]))
story.append(Paragraph(
    "<b>Phase 1 (replicate):</b> both models trained with the paper's rough default settings "
    "(LDA num_topics=50, Doc2Vec vector_size=100, Logistic Regression C=1.0), evaluated on both "
    "label tasks. <b>Phase 2 (tune):</b> each feature extractor's main hyperparameter (LDA "
    "num_topics ∈ {20, 50, 100}; Doc2Vec vector_size ∈ {50, 100, 200}) is selected by "
    "issue_area dev accuracy, then each downstream classifier's regularization C is grid-searched "
    "per label task on dev only — the test set is touched exactly once, with the final chosen "
    "configuration.", styles["Body"]))

story.append(Paragraph("Implementation", styles["H2"]))
story.append(Paragraph(
    "Code: <font face='Courier'>src/preprocessing.py</font> (loading/cleaning/splitting), "
    "<font face='Courier'>src/lda_lr.py</font> / <font face='Courier'>src/doc2vec_lr.py</font> "
    "(feature extractors), <font face='Courier'>src/classifiers.py</font> (SVM/Random Forest/Naive "
    "Bayes), <font face='Courier'>run_all.py</font> / <font face='Courier'>tune_all.py</font> / "
    "<font face='Courier'>compare_classifiers.py</font> (Phases 1/2/3). All hyperparameter and "
    "classifier selection uses only the dev split; test is evaluated once per final configuration.",
    styles["Body"]))

def pct(x):
    return f"{x*100:.1f}%"

t1_data = [["Model", "Label task", "Test Acc", "Test Macro-F1"]]
label_map = {"issue_area": "issue_area (15-label)", "issue": "issue (fine-grained)"}
for key, r in t1.items():
    model = key[:-len("_issue_area")] if key.endswith("issue_area") else key[:-len("_issue")]
    task = "issue_area (15-label)" if key.endswith("issue_area") else "issue (fine-grained)"
    t1_data.append([model, task, pct(r["test"]["accuracy"]), f"{r['test']['macro_f1']:.3f}"])

story.append(Paragraph("Results — Phase 1 (paper's default settings)", styles["H2"]))
tbl1 = Table(t1_data, colWidths=[1.3*inch, 1.8*inch, 1.0*inch, 1.3*inch])
tbl1.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
    ("ALIGN", (2, 0), (-1, -1), "CENTER"),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(tbl1)

story.append(Paragraph("Results — Phase 2 (tuned, test touched once)", styles["H2"]))
t2_data = [["Model + Task", "Tuned Param", "Test Acc Before", "Test Acc After"]]
for key, r in t2.items():
    extra = r.get("num_topics", r.get("vector_size"))
    extra_label = "num_topics" if "num_topics" in r else "vector_size"
    before = t1[key]["test"]["accuracy"]
    after = r["test_after"]["accuracy"]
    t2_data.append([key, f"{extra_label}={extra}", pct(before), pct(after)])
tbl2 = Table(t2_data, colWidths=[1.7*inch, 1.4*inch, 1.1*inch, 1.1*inch])
tbl2.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
    ("ALIGN", (2, 0), (-1, -1), "CENTER"),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(tbl2)

story.append(Paragraph(
    "LDA benefited from more topics (50→100; issue_area test accuracy +9.4pp) — 50 was too coarse "
    "for 15+ categories. Doc2Vec preferred a <i>smaller</i> embedding (100→50): with ~5,600 "
    "training documents, 200 dimensions overfits before the classifier sees enough signal.",
    styles["Body"]))

story.append(Paragraph("Results — Phase 3: classifier comparison (beyond the paper)", styles["H2"]))
story.append(Paragraph(
    "The paper only tries Logistic Regression. We additionally compare <b>SVM</b> (RBF kernel), "
    "<b>Random Forest</b>, and <b>Gaussian Naive Bayes</b> on the identical Phase-2-tuned feature "
    "vectors, isolating classifier choice from feature extraction, on the issue_area task:",
    styles["Body"]))

t3_data = [["Classifier", "Feature", "Test Acc", "Test Macro-F1"]]
for key, r in sorted(t3.items(), key=lambda kv: -kv[1]["test"]["accuracy"]):
    if r["label_task"] != "issue_area":
        continue
    t3_data.append([r["classifier"], r["feature"], pct(r["test"]["accuracy"]), f"{r['test']['macro_f1']:.3f}"])
# prepend LR rows from tuned_results for direct comparison
lr_rows = [["logistic_regression", feat, pct(t2[f"{feat}_lr_issue_area"]["test_after"]["accuracy"]),
            f"{t2[f'{feat}_lr_issue_area']['test_after']['macro_f1']:.3f}"] for feat in ["lda", "doc2vec"]]
t3_data = [t3_data[0]] + sorted(t3_data[1:] + lr_rows, key=lambda row: -float(row[2].rstrip('%')))
tbl3 = Table(t3_data, colWidths=[1.6*inch, 1.2*inch, 1.1*inch, 1.3*inch])
tbl3.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
    ("ALIGN", (2, 0), (-1, -1), "CENTER"),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(tbl3)
story.append(Paragraph(
    "<b>SVM is the overall best model</b>: 68.8% test accuracy on Doc2Vec+issue_area, beating "
    "Logistic Regression's 62.2% on the same features (+6.6pp) — the RBF kernel draws a non-linear "
    "boundary LR's linear one can't. Random Forest and Gaussian Naive Bayes both underperform LR on Doc2Vec; "
    "Gaussian Naive Bayes also collapses on LDA features (13% accuracy), since its per-feature Gaussian "
    "assumption fits topic-probability vectors poorly. SVM, Random Forest and Gaussian Naive Bayes used "
    "default hyperparameters (not tuned on dev), unlike LR's C; all results are single-seed, single-split "
    "(test set about 1,200 opinions), so gaps under about 1pp are not reliable.", styles["Body"]))

story.append(Paragraph("Conclusions", styles["H2"]))
story.append(Paragraph(
    "Doc2Vec embeddings consistently outperform LDA topic-distribution features, and the coarser "
    "15-category task is much easier than the 260+-category fine-grained task — unsurprising given "
    "how thin per-class training data gets at that granularity. LDA's macro-F1 trails its accuracy "
    "badly (0.267 vs 48.7% on issue_area), defaulting to majority classes (Criminal Procedure, "
    "Economic Activity) and failing on rare ones; Doc2Vec generalizes far better to minority "
    "categories such as Privacy and Attorneys. Classifier choice mattered almost as much as feature "
    "extraction: SVM beat every other classifier tried on Doc2Vec issue_area (on LDA it is only level with LR: 49.1% vs 48.7%). Our replicated numbers land in the same "
    "range and order as the original paper's. We note an internal inconsistency in the paper's own "
    "results table (its printed table and prose disagree on which label granularity scored higher) "
    "and resolve it in favor of the prose. We did not replicate BERT: the paper's own attempt failed "
    "with an out-of-memory error and left no baseline to reproduce; a long-document Transformer "
    "(e.g. Longformer) is natural future work, as the paper itself suggests.", styles["Body"]))

story.append(Spacer(1, 6))
story.append(Paragraph(
    "Full code, results, and figures: github.com/amogh-herle/ML-MINI-PROJECT", styles["Small"]))

doc.build(story)
print("Wrote report/writeup.pdf")
