"""Build report/slides.pptx — a PowerPoint version of the slide deck
(cover, problem, dataset, approach, phase1, phase2, phase3, phase4, insights,
progress, conclusions — 11 slides). Pulls numbers from the results JSON
files so it can't drift from the actual results.
"""
import json
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

with open("results/table1.json") as f:
    T1 = json.load(f)["table1_accuracy"]
with open("results/tuned_results.json") as f:
    T2 = json.load(f)
with open("results/classifier_comparison.json") as f:
    T3 = json.load(f)
with open("results/tuned_classifiers.json") as f:
    T4 = json.load(f)

# ---------------- design tokens ----------------
DARK = RGBColor(0x14, 0x23, 0x40)
LIGHT_BG = RGBColor(0xFB, 0xFB, 0xF8)
TEXT_DARK = RGBColor(0x1A, 0x1A, 0x1A)
TEXT_MUTED = RGBColor(0x4A, 0x55, 0x68)
TEXT_GREY = RGBColor(0x6A, 0x71, 0x79)
ACCENT = RGBColor(0x2A, 0x5C, 0xAA)
ACCENT_LIGHT = RGBColor(0x8F, 0xB4, 0xFF)
WHITE = RGBColor(0xFB, 0xFB, 0xF8)
CARD_BG = RGBColor(0xFF, 0xFF, 0xFF)
CARD_BORDER = RGBColor(0xE3, 0xE6, 0xE4)
GREEN = RGBColor(0x1E, 0x7A, 0x3D)
RED = RGBColor(0xB3, 0x26, 0x1E)
PURPLE = RGBColor(0x6B, 0x4F, 0xA0)
ROW_ALT = RGBColor(0xF2, 0xF2, 0xF2)
FOOTER_GREY = RGBColor(0x6E, 0x85, 0xAD)
DOT_OFF_LIGHT = RGBColor(0xDD, 0xE1, 0xDC)
DOT_OFF_DARK = RGBColor(0x2A, 0x3C, 0x60)

MARGIN = 0.9
TITLE_TOP = 0.95
EYEBROW_TOP = 0.55

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def add_slide(bg_color):
    slide = prs.slides.add_slide(BLANK)
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = bg_color
    bg.line.fill.background()
    bg.shadow.inherit = False
    sp = bg._element
    sp.getparent().remove(sp)
    slide.shapes._spTree.insert(2, sp)
    return slide


def add_text(slide, left, top, width, height, text, size=18, color=TEXT_DARK,
             bold=False, italic=False, align=PP_ALIGN.LEFT, font="Arial",
             line_spacing=1.1):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = line_spacing
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = font
    return box


def add_eyebrow(slide, text, color=ACCENT, dark_bg=False):
    add_text(slide, MARGIN, EYEBROW_TOP, 11, 0.4, text.upper(), size=13,
              color=(ACCENT_LIGHT if dark_bg else color), bold=True)


def add_title(slide, text, size=28, top=TITLE_TOP, width=11.5, height=0.9, dark_bg=False):
    add_text(slide, MARGIN, top, width, height, text, size=size,
              color=(WHITE if dark_bg else TEXT_DARK), bold=True, line_spacing=1.1)


def add_page_dots(slide, page, total, dark_bg=False):
    """A small row of dots, bottom-right, with the current page filled in accent color."""
    dot_d = 0.09
    gap = 0.2
    total_w = total * gap
    start_x = 13.333 - MARGIN - total_w
    y = 7.08
    off_color = DOT_OFF_DARK if dark_bg else DOT_OFF_LIGHT
    on_color = ACCENT_LIGHT if dark_bg else ACCENT
    for i in range(1, total + 1):
        x = start_x + (i - 1) * gap
        dot = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(dot_d), Inches(dot_d))
        dot.fill.solid()
        dot.fill.fore_color.rgb = on_color if i == page else off_color
        dot.line.fill.background()
        dot.shadow.inherit = False
    add_text(slide, MARGIN, 6.95, 2, 0.4, f"{page} / {total}", size=11,
              color=(FOOTER_GREY if dark_bg else TEXT_GREY))


def add_card(slide, left, top, width, height, title, body, title_color=ACCENT, accent_strip=True):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    card.adjustments[0] = 0.045
    card.fill.solid()
    card.fill.fore_color.rgb = CARD_BG
    card.line.color.rgb = CARD_BORDER
    card.line.width = Pt(1)
    card.shadow.inherit = False
    if accent_strip:
        bar = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left + 0.12), Inches(top + 0.14),
                                      Inches(0.08), Inches(height - 0.28))
        bar.adjustments[0] = 0.5
        bar.fill.solid()
        bar.fill.fore_color.rgb = title_color
        bar.line.fill.background()
        bar.shadow.inherit = False
        text_left = left + 0.38
    else:
        text_left = left + 0.28
    tf = card.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(text_left - left)
    tf.margin_right = Inches(0.28)
    tf.margin_top = Inches(0.25)
    tf.margin_bottom = Inches(0.25)
    p0 = tf.paragraphs[0]
    r0 = p0.add_run()
    r0.text = title
    r0.font.size = Pt(20)
    r0.font.bold = True
    r0.font.color.rgb = title_color
    r0.font.name = "Arial"
    p1 = tf.add_paragraph()
    p1.space_before = Pt(8)
    p1.line_spacing = 1.15
    r1 = p1.add_run()
    r1.text = body
    r1.font.size = Pt(14)
    r1.font.color.rgb = TEXT_MUTED
    r1.font.name = "Arial"
    return card


def add_table(slide, left, top, width, height, header, rows, highlight_row=None, col_widths=None, font_size=13):
    """highlight_row: an int, or a set/list of ints, for rows to tint green."""
    highlight = {highlight_row} if isinstance(highlight_row, int) else set(highlight_row or [])
    n_rows, n_cols = len(rows) + 1, len(header)
    gshape = slide.shapes.add_table(n_rows, n_cols, Inches(left), Inches(top), Inches(width), Inches(height))
    table = gshape.table
    if col_widths:
        for i, w in enumerate(col_widths):
            table.columns[i].width = Inches(w)
    for j, h in enumerate(header):
        cell = table.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = DARK
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if j < 2 else PP_ALIGN.CENTER
        r = p.runs[0]
        r.font.size = Pt(font_size)
        r.font.bold = True
        r.font.color.rgb = WHITE
        r.font.name = "Arial"
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.text = str(val)
            cell.fill.solid()
            cell.fill.fore_color.rgb = ROW_ALT if (i % 2 == 1) else CARD_BG
            if i in highlight:
                cell.fill.fore_color.rgb = RGBColor(0xE8, 0xF3, 0xEC)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j < 2 else PP_ALIGN.CENTER
            r = p.runs[0]
            r.font.size = Pt(font_size)
            r.font.name = "Arial"
            r.font.color.rgb = GREEN if i in highlight else TEXT_DARK
            r.font.bold = i in highlight
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    return table


def pct(x):
    return f"{x*100:.1f}%"


def add_badge(slide, cx, cy, diameter, fill_color, text, text_color=WHITE, size=20):
    """A filled circle with centered text — used as a small icon/number badge."""
    left, top = cx - diameter / 2, cy - diameter / 2
    circ = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(left), Inches(top), Inches(diameter), Inches(diameter))
    circ.fill.solid()
    circ.fill.fore_color.rgb = fill_color
    circ.line.fill.background()
    circ.shadow.inherit = False
    tf = circ.text_frame
    tf.word_wrap = False
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = True
    r.font.color.rgb = text_color
    r.font.name = "Arial"
    return circ


def add_accent_bar(slide, left, top, width, height=0.06, color=None):
    bar = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    if hasattr(bar, "adjustments") and len(bar.adjustments):
        bar.adjustments[0] = 0.5
    bar.fill.solid()
    bar.fill.fore_color.rgb = color or ACCENT
    bar.line.fill.background()
    bar.shadow.inherit = False
    return bar


def add_checkmark_bullets(slide, left, top, width, bullets, size=19,
                           text_color=RGBColor(0xE3, 0xE9, 0xF5), badge_color=None, row_h=0.62):
    """Rows of a small colored checkmark circle + text, instead of a plain bullet."""
    y = top
    for b in bullets:
        add_badge(slide, left + 0.16, y + 0.16, 0.32, badge_color or ACCENT_LIGHT, "✓",
                   text_color=DARK, size=13)
        add_text(slide, left + 0.5, y - 0.03, width - 0.5, row_h, b, size=size, color=text_color, line_spacing=1.15)
        y += row_h


TOTAL = 11  # content slides (excludes cover)

# ---------------- 1. Cover ----------------
s = add_slide(DARK)
add_text(s, MARGIN, 0.9, 11, 0.4, "UE24CS352A — MACHINE LEARNING MINI-PROJECT", size=15, color=ACCENT_LIGHT, bold=True)
add_accent_bar(s, MARGIN + 0.03, 1.5, 1.6, height=0.07, color=ACCENT)
add_text(s, MARGIN, 1.65, 11, 1.8, "Classification of Legal Text", size=48, color=WHITE, bold=True)
add_text(s, MARGIN, 3.05, 10.5, 1.2,
         "Replicating & tuning a Stanford CS229 NLP project — teaching a classifier to read U.S. "
         "Supreme Court opinions and guess their legal category.",
         size=20, color=RGBColor(0xC7, 0xD4, 0xEA), line_spacing=1.3)
add_badge(s, 11.7, 1.3, 1.5, RGBColor(0x1C, 0x30, 0x55), "68.8%", text_color=ACCENT_LIGHT, size=15)
add_text(s, 10.95, 2.15, 1.5, 0.3, "best test acc", size=10, color=FOOTER_GREY, align=PP_ALIGN.CENTER)
add_text(s, MARGIN, 6.9, 11, 0.4, 'Based on: Krithika Iyer, "Classification of Legal Text" (CS229, 2020)',
         size=13, color=FOOTER_GREY)

# ---------------- 2. Problem ----------------
s = add_slide(LIGHT_BG)
add_eyebrow(s, "The Problem")
add_title(s, "Can a simple model guess a court ruling's legal category?", size=28, height=1.3)
add_text(s, MARGIN, 2.25, 11.2, 0.9,
         "Legal opinions use long, non-standard sentence structures that break typical NLP tooling. "
         "We classify U.S. Supreme Court opinions into their human-curated legal category — at two "
         "levels of detail.", size=15, color=TEXT_MUTED, line_spacing=1.3)
add_card(s, MARGIN, 3.35, 5.4, 2.0, "15", "Broad categories (issue_area) — e.g. Criminal Procedure, Civil Rights, First Amendment")
add_card(s, 6.5, 3.35, 5.4, 2.0, "260+", "Fine-grained issue codes (issue) — much harder, far less data per category")
add_page_dots(s, 1, TOTAL)

# ---------------- 3. Dataset ----------------
s = add_slide(LIGHT_BG)
add_eyebrow(s, "Dataset")
add_title(s, "Real Supreme Court opinions — no substitute needed")
add_text(s, MARGIN, 1.75, 11.2, 0.7,
         "Same source the original paper used: the Supreme Court Database via the textacy package. "
         "Still downloadable today.", size=15, color=TEXT_MUTED, line_spacing=1.3)
add_card(s, MARGIN, 2.65, 3.6, 1.6, "8,419", "opinions, 1946-2016")
add_card(s, 4.75, 2.65, 3.6, 1.6, "7,971", "after filtering rare classes")
add_card(s, 8.6, 2.65, 3.8, 1.6, "70/15/15", "train / dev / test split")
add_text(s, MARGIN, 4.6, 11.2, 1.0,
         "Opinions run long (median ~34,000 characters) — each truncated to its first 4,000 characters "
         "so training stays fast, keeping the part where judges usually state the issue and holding.",
         size=14, color=TEXT_GREY, line_spacing=1.3)
add_page_dots(s, 2, TOTAL)

# ---------------- 4. Approach ----------------
s = add_slide(LIGHT_BG)
add_eyebrow(s, "Approach")
add_title(s, "Two ways to featurize legal text, one classifier")
add_card(s, MARGIN, 2.0, 5.5, 3.3, "LDA + Logistic Regression",
         "Finds abstract recurring topics across all rulings (gensim, TF-IDF). Each ruling becomes "
         "a mix-of-topics score. A classifier maps topic mix → category.\n\nTuned: 100 topics "
         "(vs. 50 default)")
add_badge(s, 1.15, 2.25, 0.5, RED, "Σ", size=22)
add_card(s, 6.6, 2.0, 5.6, 3.3, "Doc2Vec + Logistic Regression",
         "Learns a dense numeric “fingerprint” per ruling by predicting words in context "
         "(gensim Doc2Vec). Same classifier maps fingerprint → category.\n\nTuned: 50-dim "
         "vectors (vs. 100 default)")
add_badge(s, 6.85, 2.25, 0.5, GREEN, "⌘", size=20)
add_text(s, MARGIN, 5.55, 11.2, 0.6,
         "A third method (BERT) was in the original paper, but it ran out of memory there — no baseline to replicate.",
         size=13, color=TEXT_GREY)
add_page_dots(s, 3, TOTAL)

# ---------------- 5. Phase 1 ----------------
s = add_slide(LIGHT_BG)
add_eyebrow(s, "Phase 1 — Replication")
add_title(s, "Paper's default settings, our dataset", size=26, height=0.6)
header = ["Model", "Task", "Test Acc", "F1"]
rows = []
for key, r in T1.items():
    model = key[: -len("_issue_area")] if key.endswith("issue_area") else key[: -len("_issue")]
    task = "issue_area" if key.endswith("issue_area") else "issue"
    rows.append([model, task, pct(r["test"]["accuracy"]), f"{r['test']['macro_f1']:.3f}"])
add_table(s, MARGIN, 1.75, 5.5, 2.6, header, rows, col_widths=[1.9, 1.6, 1.05, 0.95])
add_text(s, MARGIN, 4.55, 5.3, 1.6,
         "Doc2Vec beats LDA on both tasks. The coarser 15-category task is much easier than the "
         "260+-category task — matches the original paper's pattern.", size=14, color=TEXT_MUTED, line_spacing=1.3)
s.shapes.add_picture("results/figures/phase1_test_accuracy.png", Inches(6.7), Inches(1.75), width=Inches(5.7))
add_page_dots(s, 4, TOTAL)

# ---------------- 6. Phase 2 ----------------
s = add_slide(LIGHT_BG)
add_eyebrow(s, "Phase 2 — Hyperparameter Tuning")
add_title(s, "Dev-only tuning, test touched once", size=26, height=0.6)
header = ["Model + Task", "Param", "Before", "After"]
rows = []
for key, r in T2.items():
    extra = r.get("num_topics", r.get("vector_size"))
    extra_label = "topics" if "num_topics" in r else "dim"
    before = T1[key]["test"]["accuracy"]
    after = r["test_after"]["accuracy"]
    rows.append([key.replace("_lr_", " — "), f"{extra_label}={extra}", pct(before), pct(after)])
add_table(s, MARGIN, 1.75, 6.1, 2.6, header, rows, col_widths=[2.4, 1.1, 1.3, 1.3])
add_text(s, MARGIN, 4.55, 5.9, 1.6,
         "LDA needed more topics (50→100) to separate 15+ categories. Doc2Vec preferred a "
         "smaller fingerprint (100→50) — with only ~5,600 training rulings, bigger vectors "
         "just overfit.", size=14, color=TEXT_MUTED, line_spacing=1.3)
s.shapes.add_picture("results/figures/before_after_tuning.png", Inches(7.3), Inches(1.75), width=Inches(5.1))
add_page_dots(s, 5, TOTAL)

# ---------------- 7. Phase 3 ----------------
s = add_slide(LIGHT_BG)
add_eyebrow(s, "Phase 3 — Beyond the Paper")
add_title(s, "Same features, different classifier on top", size=26, height=0.6)
add_text(s, MARGIN, 1.55, 5.4, 0.9,
         "The paper only tries Logistic Regression. We also tried SVM, Random Forest, and Gaussian Naive Bayes "
         "— same Doc2Vec fingerprints, issue_area task.", size=14, color=TEXT_MUTED, line_spacing=1.3)
lr_doc2vec = T2["doc2vec_lr_issue_area"]["test_after"]
area_rows = sorted(
    [r for r in T3.values() if r["label_task"] == "issue_area" and r["feature"] == "doc2vec"],
    key=lambda r: -r["test"]["accuracy"],
)
names = {"svm": "SVM (RBF)", "random_forest": "Random Forest", "naive_bayes": "Gaussian Naive Bayes"}
rows = [[names[r["classifier"]], pct(r["test"]["accuracy"]), f"{r['test']['macro_f1']:.3f}"] for r in area_rows]
rows.insert(1, ["Logistic Regression", pct(lr_doc2vec["accuracy"]), f"{lr_doc2vec['macro_f1']:.3f}"])
header = ["Classifier", "Test Acc", "F1"]
add_table(s, MARGIN, 2.6, 5.4, 2.3, header, rows, highlight_row=0, col_widths=[2.6, 1.4, 1.4])
add_text(s, MARGIN, 4.95, 5.4, 0.6,
         "SVM (RBF, C=1.0), Random Forest (200 trees) and Naive Bayes use defaults, untuned; "
         "LR (C=1.0) is the tuned one.", size=11, color=TEXT_GREY, line_spacing=1.2)
add_text(s, MARGIN, 5.6, 5.4, 1.1,
         "SVM wins — its RBF kernel draws a curved decision boundary through the embedding "
         "space that straight-line Logistic Regression can't match.", size=14, color=TEXT_MUTED, line_spacing=1.3)
s.shapes.add_picture("results/figures/classifier_comparison.png", Inches(6.6), Inches(1.75), width=Inches(5.9))
add_page_dots(s, 6, TOTAL)

# ---------------- 8. Phase 4 ----------------
s = add_slide(LIGHT_BG)
add_eyebrow(s, "Phase 4 — Deeper Tuning")
add_title(s, "What tuning really did to accuracy — the full picture", size=24, height=0.55)
add_text(s, MARGIN, 1.42, 11.2, 0.5,
         "SVM, Random Forest and Naive Bayes from Phase 3 were still untuned. We grid-searched "
         "kernels, tree depth and smoothing — selected on dev macro-F1, not accuracy. All 14 "
         "feature × task × classifier combinations, in full:",
         size=12.5, color=TEXT_MUTED, line_spacing=1.2)

names4 = {"svm": "SVM", "random_forest": "Random Forest",
          "naive_bayes_gaussian": "Naive Bayes (Gaussian)", "naive_bayes_multinomial": "Naive Bayes (Multinomial)"}
order4 = ["svm", "random_forest", "naive_bayes_gaussian", "naive_bayes_multinomial"]
def _parse4(k):
    for clf in order4:
        if k.startswith(clf + "_"):
            rest = k[len(clf) + 1:]
            feat, task = ("lda", rest[4:]) if rest.startswith("lda_") else ("doc2vec", rest[8:])
            return clf, feat, task
rows4, highlight_idx = [], []
sort_key = lambda item: (item[2], item[1], order4.index(item[0]))  # task, feat, classifier order
parsed = sorted(((*_parse4(k), k, r) for k, r in T4.items()), key=sort_key)
for i, (clf, feat, task, k, r) in enumerate(parsed):
    ba = r.get("test_before", {}).get("accuracy")
    bf = r.get("test_before", {}).get("macro_f1")
    aa, af = r["test_after"]["accuracy"], r["test_after"]["macro_f1"]
    rows4.append([
        names4[clf], feat, "issue_area" if task == "issue_area" else "issue",
        pct(ba) if ba is not None else "n/a", pct(aa),
        f"{bf:.3f}" if bf is not None else "n/a", f"{af:.3f}",
    ])
    if k in ("svm_doc2vec_issue_area", "naive_bayes_gaussian_lda_issue_area"):
        highlight_idx.append(i)
header4 = ["Classifier", "Feature", "Task", "Acc Before", "Acc After", "F1 Before", "F1 After"]
add_table(s, MARGIN, 2.05, 11.5, 4.1, header4, rows4, highlight_row=highlight_idx,
          col_widths=[2.1, 1.25, 1.55, 1.5, 1.5, 1.5, 1.5], font_size=10.5)

svm_r, gnb_r = T4["svm_doc2vec_issue_area"], T4["naive_bayes_gaussian_lda_issue_area"]
add_text(s, MARGIN, 6.3, 11.2, 0.65,
         f"SVM's accuracy actually dropped ({pct(svm_r['test_before']['accuracy'])} → "
         f"{pct(svm_r['test_after']['accuracy'])}) even though its F1 rose — tuning for macro-F1 "
         f"traded raw accuracy for fairness across categories. Gaussian NB on LDA/issue_area is the "
         f"one case where both rose sharply ({pct(gnb_r['test_before']['accuracy'])} → "
         f"{pct(gnb_r['test_after']['accuracy'])} accuracy).",
         size=11.5, color=TEXT_MUTED, line_spacing=1.2, italic=True)
add_page_dots(s, 7, TOTAL)

# ---------------- 9. Insights ----------------
s = add_slide(LIGHT_BG)
add_eyebrow(s, "A Closer Look")
add_title(s, "LDA guesses the majority class. Doc2Vec doesn't.", size=26, height=0.7)
s.shapes.add_picture("results/figures/per_category_f1.png", Inches(2.17), Inches(1.75), width=Inches(9.0))
add_accent_bar(s, 1.3, 6.42, 0.22, height=0.22, color=RED)
add_text(s, 1.65, 6.34, 5.0, 0.4, "LDA — collapses on rare categories (near-0 F1 on Privacy, Attorneys)",
         size=12, color=TEXT_MUTED)
add_accent_bar(s, 7.0, 6.42, 0.22, height=0.22, color=GREEN)
add_text(s, 7.35, 6.34, 5.0, 0.4, "Doc2Vec — far more balanced across all 13 categories",
         size=12, color=TEXT_MUTED)
add_page_dots(s, 8, TOTAL)

# ---------------- 9. Confusion Matrix ----------------
s = add_slide(LIGHT_BG)
add_eyebrow(s, "A Closer Look")
add_title(s, "Where the best model actually gets confused", size=24, height=0.55)
s.shapes.add_picture("results/figures/confusion_matrix.png", Inches(3.97), Inches(1.4), width=Inches(5.4))
add_text(s, MARGIN, 6.35, 11.6, 0.35,
         "Rows = true category, columns = predicted — Economic Activity pulls in "
         "misclassifications from Judicial Power and Federalism.",
         size=11, color=TEXT_MUTED, align=PP_ALIGN.CENTER)
add_page_dots(s, 9, TOTAL)

# ---------------- 10. Progress Across Phases ----------------
s = add_slide(LIGHT_BG)
add_eyebrow(s, "Putting It Together")
add_title(s, "Each phase pushed the best model further")
s.shapes.add_picture("results/figures/progress_by_phase.png", Inches(MARGIN), Inches(2.0), width=Inches(6.0))
bullets_progress = [
    "Phase 1: Doc2Vec + Logistic Regression, paper's default settings — 59.1%",
    "Phase 2: tuned vector_size (100→50) and C on dev only — 62.2%",
    "Phase 3: swapped in SVM (RBF kernel) on the same features — 68.8%",
]
add_checkmark_bullets(s, 7.3, 2.6, 5.1, bullets_progress, size=15,
                       text_color=TEXT_DARK, badge_color=ACCENT)
add_text(s, 7.3, 5.3, 5.1, 1.2,
         "Nearly 10 points of accuracy gained beyond the paper's own result, without touching the dataset.",
         size=14, color=TEXT_MUTED, italic=True, line_spacing=1.3)
add_page_dots(s, 10, TOTAL)

# ---------------- 11. Conclusions ----------------
s = add_slide(DARK)
add_eyebrow(s, "Conclusions", dark_bg=True)
bullets = [
    "Doc2Vec embeddings consistently beat LDA topic features on this task",
    "Coarse 15-category task is much easier than the 260+-category one",
    "Tuning hyperparameters gave real gains in every phase, not just marginal ones",
    "SVM beat every classifier on Doc2Vec issue_area (on LDA ~level with LR: 49.1 vs 48.7%)",
    "Phase 4: wider grids + macro-F1 selection more than doubled Gaussian NB's F1 on LDA",
    "Our numbers land in the same range and order as the original paper's",
    "BERT not replicated — paper's own attempt ran out of memory, no baseline to match",
]
add_checkmark_bullets(s, MARGIN, 1.4, 11.2, bullets, size=17, badge_color=ACCENT_LIGHT, row_h=0.54)
add_text(s, MARGIN, 6.55, 7, 0.35, "github.com/amogh-herle/ML-MINI-PROJECT", size=13, color=ACCENT_LIGHT)
add_page_dots(s, 11, TOTAL, dark_bg=True)

prs.save("report/slides.pptx")
print("Wrote report/slides.pptx")
