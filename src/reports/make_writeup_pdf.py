"""Build the 2-page write-up PDF required by the assignment guidelines
(problem statement, dataset details, approach, implementation overview,
conclusions). Pulls numbers from the results JSON files so the write-up
never drifts from the actual results.
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
with open("results/tuned_classifiers.json") as f:
    t4 = json.load(f)
with open("results/sbert_results.json") as f:
    t5 = json.load(f)


def pct(x):
    return f"{x*100:.1f}%"


p3_svm_doc2vec_area = next(
    r for r in t3.values()
    if r["classifier"] == "svm" and r["feature"] == "doc2vec" and r["label_task"] == "issue_area"
)
p5_sbert_svm_area = t5["sbert_svm_issue_area"]["test"]

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleCustom", parent=styles["Title"], fontSize=16, spaceAfter=2))
styles.add(ParagraphStyle(name="SubTitle", parent=styles["Normal"], fontSize=10, textColor=colors.grey, spaceAfter=10))
styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], fontSize=12, spaceBefore=7, spaceAfter=3))
styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"], fontSize=9, leading=11.8, spaceAfter=4))
styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10, textColor=colors.grey))

TABLE_STYLE = TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 8),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("TOPPADDING", (0, 0), (-1, -1), 2.5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
])

doc = SimpleDocTemplate(
    "report/writeup.pdf", pagesize=LETTER,
    topMargin=0.5 * inch, bottomMargin=0.5 * inch,
    leftMargin=0.6 * inch, rightMargin=0.6 * inch,
)

story = []

story.append(Paragraph("Classification of Legal Text: Replication &amp; Extension", styles["TitleCustom"]))
story.append(Paragraph(
    "Mini-Project Write-up &nbsp;|&nbsp; Replicating &amp; extending Krithika Iyer, "
    "<i>“Classification of Legal Text,”</i> Stanford CS229 (2020)", styles["SubTitle"]))

story.append(Paragraph("Problem Statement", styles["H2"]))
story.append(Paragraph(
    "Legal opinions use long, non-standard sentence structures that break typical NLP tooling. "
    "We replicate and extend a Stanford CS229 project that classifies U.S. Supreme Court opinions "
    "into their human-curated legal subject-matter category, at two granularities: <b>issue_area</b> "
    "(15 broad categories) and <b>issue</b> (260+ fine-grained codes). Beyond replication, we test "
    "whether modern pretrained embeddings beat the paper's from-scratch Doc2Vec.", styles["Body"]))

story.append(Paragraph("Dataset", styles["H2"]))
story.append(Paragraph(
    "The exact dataset the original paper used: the Supreme Court Database via the <b>textacy</b> "
    "package's bundled <b>SupremeCourt</b> dataset — 8,419 opinions (1946-2016). After dropping "
    "opinions with no issue_area label and issue codes with fewer than 10 examples, 7,971 opinions "
    "remain, split 70/15/15 into train/dev/test, stratified by issue_area. Opinions run long (median "
    "~34,000 characters); each is truncated to its first 4,000 characters before vectorization.",
    styles["Body"]))

story.append(Paragraph("Approach — Five Phases", styles["H2"]))
story.append(Paragraph(
    "<b>Features:</b> (1) LDA — gensim topic model, TF-IDF, no training beyond fitting the topic "
    "distribution. (2) Doc2Vec — a small neural net, trained from scratch on our ~5,600 training "
    "documents only. (3) SBERT (all-MiniLM-L6-v2) — a transformer pretrained on 1B+ sentence pairs; "
    "used only to encode, never trained. <b>Classifiers:</b> Logistic Regression (the paper's), plus "
    "SVM, Random Forest, and Gaussian Naive Bayes.", styles["Body"]))
story.append(Paragraph(
    "<b>Phase 1</b> replicates the paper's Doc2Vec+LR with its default settings. <b>Phase 2</b> tunes "
    "each feature extractor's main hyperparameter (LDA num_topics, Doc2Vec vector_size) and each "
    "classifier's C, selected on dev. <b>Phase 3</b> compares SVM/Random Forest/Naive Bayes against "
    "LR on the same tuned features (the paper never does this). <b>Phase 4</b> deep-tunes just the "
    "single best model from Phase 3 (SVM + Doc2Vec + issue_area) over 40 kernel/C/gamma combinations, "
    "selecting on dev <i>accuracy</i> specifically. <b>Phase 5</b> swaps Doc2Vec for SBERT — zero "
    "training for the embedding step — on the same classifiers and tasks. All hyperparameter and "
    "classifier selection uses only the dev split; test is evaluated once per final configuration.",
    styles["Body"]))

story.append(Paragraph("Implementation", styles["H2"]))
story.append(Paragraph(
    "<font face='Courier'>src/data/</font> (corpus loading, splitting, labels), "
    "<font face='Courier'>src/models/</font> (lda_lr, doc2vec_lr, sbert_lr, classifiers), "
    "<font face='Courier'>scripts/01_replicate.py</font> … <font face='Courier'>05_sbert.py</font> "
    "(the five phases, in order), <font face='Courier'>scripts/train_best_model.py</font> + "
    "<font face='Courier'>predict.py</font> (saves the best model, classifies new text for a live "
    "demo), <font face='Courier'>scripts/evaluate_best_model.py</font> (confusion matrix).",
    styles["Body"]))

story.append(Paragraph("Results — Progress Across All Five Phases (issue_area)", styles["H2"]))
progress_data = [
    ["Phase", "Model", "Test Acc", "Test Macro-F1"],
    ["1 — Replicate", "Doc2Vec + LR (paper defaults)", pct(t1["doc2vec_lr_issue_area"]["test"]["accuracy"]),
     f"{t1['doc2vec_lr_issue_area']['test']['macro_f1']:.3f}"],
    ["2 — Tune features", "Doc2Vec + LR (tuned vector_size, C)", pct(t2["doc2vec_lr_issue_area"]["test_after"]["accuracy"]),
     f"{t2['doc2vec_lr_issue_area']['test_after']['macro_f1']:.3f}"],
    ["3 — Compare classifiers", "Doc2Vec + SVM (default)", pct(p3_svm_doc2vec_area["test"]["accuracy"]),
     f"{p3_svm_doc2vec_area['test']['macro_f1']:.3f}"],
    ["4 — Deep-tune winner", f"Doc2Vec + SVM ({t4['best_params']['kernel']})", pct(t4["test_after"]["accuracy"]),
     f"{t4['test_after']['macro_f1']:.3f}"],
    ["5 — Pretrained embeddings", "SBERT + SVM", pct(p5_sbert_svm_area["accuracy"]),
     f"{p5_sbert_svm_area['macro_f1']:.3f}"],
]
tbl = Table(progress_data, colWidths=[1.5*inch, 2.3*inch, 1.0*inch, 1.3*inch])
tbl.setStyle(TABLE_STYLE)
story.append(tbl)
story.append(Paragraph(
    f"Phase 4 selected on dev <i>accuracy</i> (not macro-F1, unlike earlier phases) specifically "
    f"because the deployed model's accuracy is what matters in practice; even so, test accuracy held "
    f"flat ({pct(t4['test_before']['accuracy'])}→{pct(t4['test_after']['accuracy'])}, within noise "
    f"for an ~850-example test set) while macro-F1 rose "
    f"({t4['test_before']['macro_f1']:.3f}→{t4['test_after']['macro_f1']:.3f}) via a kernel change "
    f"(RBF→{t4['best_params']['kernel']}). Phase 5's SBERT then beat every Doc2Vec variant at "
    f"every feature/task/classifier combination we tried — a modern pretrained model, with zero "
    f"training, outperforms a model trained from scratch on our small corpus.", styles["Body"]))

story.append(Paragraph("Per-Category Pattern (issue_area)", styles["H2"]))
story.append(Paragraph(
    "LDA's macro-F1 trails its accuracy badly, defaulting to majority classes (Criminal Procedure, "
    "Economic Activity) and collapsing to near-zero F1 on minority ones (Privacy, Attorneys). Doc2Vec "
    "and SBERT both generalize far better to minority categories, since they capture context rather "
    "than just word co-occurrence. Error analysis shows the best model reads surface vocabulary, not "
    "legal reasoning: an Attorneys fee-dispute opinion dominated by bond/surety financial language was "
    "misclassified as Federal Taxation (97% confidence), while an opinion with explicit labor-law "
    "vocabulary (“Secretary of Labor,” “union election”) was correctly classified as Unions.",
    styles["Body"]))

story.append(Paragraph("Conclusions &amp; Limitations", styles["H2"]))
story.append(Paragraph(
    f"Pretrained embeddings beat from-scratch ones at every comparison point; classifier choice "
    f"matters almost as much as features (SVM beat every other classifier tried); tuning helped in "
    f"every phase, though Phase 4 traded flat accuracy for a macro-F1 gain; the coarse 15-category "
    f"task is far easier than the 260+-category one everywhere we tried (best ~48% accuracy) — a data "
    f"scarcity problem, not a model problem. Overall: {pct(t1['doc2vec_lr_issue_area']['test']['accuracy'])} "
    f"→ {pct(p5_sbert_svm_area['accuracy'])} test accuracy across five phases, without ever "
    f"touching the dataset. Our replicated numbers land in the same range and order as the original "
    f"paper's; we also note an internal inconsistency in the paper's own results table (its printed "
    f"table and prose disagree on which label granularity scored higher) and resolve it in favor of "
    f"the prose. <b>Limitations:</b> single train/dev/test split (no cross-validation, so small "
    f"accuracy gaps carry real uncertainty); 4,000-character truncation discards most of each opinion; "
    f"fine-grained issue classification remains weak everywhere. <b>Future work:</b> a legal-domain "
    f"pretrained model (Legal-BERT) and full BERT fine-tuning (vs. SBERT's frozen embeddings) are "
    f"natural next steps, as the original paper itself suggested before its own attempt ran out of "
    f"memory.", styles["Body"]))

story.append(Spacer(1, 4))
story.append(Paragraph(
    "Full code, results, and figures: github.com/amogh-herle/ML-MINI-PROJECT", styles["Small"]))

doc.build(story)
print("Wrote report/writeup.pdf")
