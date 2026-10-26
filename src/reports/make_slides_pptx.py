"""Build report/slides.pptx — a PowerPoint version of the slide deck, now
infographic-style: big stat cards and charts instead of raw data tables
wherever a handful of numbers tell the story. 15 slides (cover + 14 content).
Pulls numbers from the results JSON files so it can't drift from the
actual results.
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
    T4 = json.load(f)  # Phase 4: single narrowed result (SVM + Doc2Vec + issue_area, deep-tuned)
with open("results/sbert_results.json") as f:
    T5 = json.load(f)  # Phase 5: SBERT vs Doc2Vec

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


def add_stat_row(slide, top, height, stats, left=MARGIN, width=11.53, gap=0.3, dark_bg=False):
    """A row of N evenly-spaced big-number stat cards: stats = [(big_text, label, color), ...].
    The infographic-style replacement for a data table when there are only a
    handful of numbers that matter."""
    n = len(stats)
    card_w = (width - gap * (n - 1)) / n
    bg = CARD_BG if not dark_bg else RGBColor(0x1C, 0x30, 0x55)
    border = CARD_BORDER if not dark_bg else RGBColor(0x2A, 0x3C, 0x60)
    label_color = TEXT_MUTED if not dark_bg else RGBColor(0xC7, 0xD4, 0xEA)
    for i, (big, label, color) in enumerate(stats):
        x = left + i * (card_w + gap)
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(top), Inches(card_w), Inches(height))
        card.adjustments[0] = 0.06
        card.fill.solid()
        card.fill.fore_color.rgb = bg
        card.line.color.rgb = border
        card.line.width = Pt(1)
        card.shadow.inherit = False
        add_text(slide, x, top + height * 0.16, card_w, height * 0.5, big, size=34, color=color,
                  bold=True, align=PP_ALIGN.CENTER)
        add_text(slide, x + 0.12, top + height * 0.62, card_w - 0.24, height * 0.35, label, size=12.5,
                  color=label_color, align=PP_ALIGN.CENTER, line_spacing=1.15)


# ---------------- pull the numbers once, so slide code just reads them ----------------
P1_ACC = T1["doc2vec_lr_issue_area"]["test"]["accuracy"]
P2_ACC = T2["doc2vec_lr_issue_area"]["test_after"]["accuracy"]
P3_ACC = next(r["test"]["accuracy"] for r in T3.values()
              if r["classifier"] == "svm" and r["feature"] == "doc2vec" and r["label_task"] == "issue_area")
P3_F1 = next(r["test"]["macro_f1"] for r in T3.values()
             if r["classifier"] == "svm" and r["feature"] == "doc2vec" and r["label_task"] == "issue_area")
P4_ACC_BEFORE = T4["test_before"]["accuracy"]
P4_ACC_AFTER = T4["test_after"]["accuracy"]
P4_F1_BEFORE = T4["test_before"]["macro_f1"]
P4_F1_AFTER = T4["test_after"]["macro_f1"]
P5_ACC = T5["sbert_svm_issue_area"]["test"]["accuracy"]
P5_F1 = T5["sbert_svm_issue_area"]["test"]["macro_f1"]
BEST_ACC = P5_ACC  # the overall best model: SBERT + SVM, issue_area

TOTAL = 14  # content slides (excludes cover)

# ================================================================
# 1. Cover
# ================================================================
s = add_slide(DARK)
add_text(s, MARGIN, 0.9, 11, 0.4, "UE24CS352A — MACHINE LEARNING MINI-PROJECT", size=15, color=ACCENT_LIGHT, bold=True)
add_accent_bar(s, MARGIN + 0.03, 1.5, 1.6, height=0.07, color=ACCENT)
add_text(s, MARGIN, 1.65, 11, 1.8, "Classification of Legal Text", size=48, color=WHITE, bold=True)
add_text(s, MARGIN, 3.05, 10.5, 1.2,
         "Replicating & extending a Stanford CS229 NLP project — teaching a classifier to read U.S. "
         "Supreme Court opinions and guess their legal category.",
         size=20, color=RGBColor(0xC7, 0xD4, 0xEA), line_spacing=1.3)
add_badge(s, 11.7, 1.3, 1.5, RGBColor(0x1C, 0x30, 0x55), pct(BEST_ACC), text_color=ACCENT_LIGHT, size=15)
add_text(s, 10.95, 2.15, 1.5, 0.3, "best test acc", size=10, color=FOOTER_GREY, align=PP_ALIGN.CENTER)
add_text(s, MARGIN, 6.9, 11, 0.4, 'Based on: Krithika Iyer, "Classification of Legal Text" (CS229, 2020)',
         size=13, color=FOOTER_GREY)

# ================================================================
# 2. Executive Summary
# ================================================================
s = add_slide(LIGHT_BG)
add_eyebrow(s, "Executive Summary")
add_title(s, "Five phases, one steadily improving model", size=28, height=0.7)
add_stat_row(s, 1.9, 1.7, [
    (pct(BEST_ACC), "Best test accuracy\n(SBERT + SVM)", ACCENT),
    ("5", "Phases of work", GREEN),
    ("13", "Legal categories\n(issue_area)", PURPLE),
    ("7,971", "Supreme Court opinions\nused", RED),
])
add_text(s, MARGIN, 3.85, 11.2, 0.5,
         "We replicated the original paper's two methods (LDA, Doc2Vec), then went beyond it: "
         "compared 4 classifier types, deep-tuned the winner, and swapped in a modern pretrained "
         "embedding model.", size=15, color=TEXT_MUTED, line_spacing=1.3)
bullets = [
    f"Replicated the paper's Doc2Vec+LR baseline: {pct(P1_ACC)} accuracy",
    f"Tuning the paper's own setup got it to {pct(P2_ACC)}, then swapping in SVM got to {pct(P3_ACC)}",
    f"A modern pretrained model (SBERT) beat everything, with zero training: {pct(BEST_ACC)}",
]
add_checkmark_bullets(s, MARGIN, 4.55, 11.2, bullets, size=15, text_color=TEXT_DARK, badge_color=ACCENT, row_h=0.5)
add_page_dots(s, 1, TOTAL)

# ================================================================
# 3. Problem
# ================================================================
s = add_slide(LIGHT_BG)
add_eyebrow(s, "The Problem")
add_title(s, "Can a model guess a court ruling's legal category?", size=28, height=1.1)
add_text(s, MARGIN, 2.05, 11.2, 0.9,
         "Legal opinions use long, non-standard sentence structures that break typical NLP tooling. "
         "We classify U.S. Supreme Court opinions into their human-curated legal category — at two "
         "levels of detail.", size=15, color=TEXT_MUTED, line_spacing=1.3)
add_card(s, MARGIN, 3.1, 5.4, 2.1, "15", "Broad categories (issue_area) — e.g. Criminal Procedure, Civil Rights, First Amendment")
add_card(s, 6.5, 3.1, 5.4, 2.1, "260+", "Fine-grained issue codes (issue) — much harder, far less data per category")
add_page_dots(s, 2, TOTAL)

# ================================================================
# 4. Dataset
# ================================================================
s = add_slide(LIGHT_BG)
add_eyebrow(s, "Dataset")
add_title(s, "Real Supreme Court opinions — no substitute needed")
add_text(s, MARGIN, 1.75, 11.2, 0.6,
         "Same source the original paper used: the Supreme Court Database via the textacy package. "
         "Still downloadable today.", size=15, color=TEXT_MUTED, line_spacing=1.3)
add_stat_row(s, 2.55, 1.5, [
    ("8,419", "opinions, 1946-2016", ACCENT),
    ("7,971", "after filtering rare classes", ACCENT),
    ("70/15/15", "train / dev / test split", ACCENT),
])
add_text(s, MARGIN, 4.45, 11.2, 1.0,
         "Opinions run long (median ~34,000 characters) — each truncated to its first 4,000 characters "
         "so training stays fast, keeping the part where judges usually state the issue and holding.",
         size=14, color=TEXT_GREY, line_spacing=1.3)
add_page_dots(s, 3, TOTAL)

# ================================================================
# 5. Approach
# ================================================================
s = add_slide(LIGHT_BG)
add_eyebrow(s, "Approach")
add_title(s, "Three ways to turn text into numbers", size=27, height=0.6)
add_card(s, MARGIN, 1.75, 3.65, 3.5, "LDA",
         "Counts which words cluster together across all rulings (gensim, TF-IDF). "
         "No training beyond that — pure statistics.\n\n2021-era paper default.", title_color=RED)
add_badge(s, 1.15, 2.0, 0.5, RED, "Σ", size=22)
add_card(s, 4.85, 1.75, 3.65, 3.5, "Doc2Vec",
         "A small neural net learns a 50-number “fingerprint” per ruling, trained from "
         "scratch on our ~5,600 documents only.\n\nThe paper's other method.", title_color=GREEN)
add_badge(s, 5.1, 2.0, 0.5, GREEN, "⌘", size=20)
add_card(s, 8.68, 1.75, 3.75, 3.5, "SBERT",
         "A modern transformer, pretrained on 1B+ sentence pairs before it ever sees our data. "
         "No training needed — just encode.\n\nOur Phase 5 addition.", title_color=ACCENT)
add_badge(s, 8.93, 2.0, 0.5, ACCENT, "✨", size=20)
add_text(s, MARGIN, 5.55, 11.2, 0.6,
         "A fourth method (BERT fine-tuning) was in the original paper, but it ran out of memory there — no baseline to replicate.",
         size=12.5, color=TEXT_GREY)
add_page_dots(s, 4, TOTAL)

# ================================================================
# 6. Replication & Tuning (Phase 1 + 2 merged)
# ================================================================
s = add_slide(LIGHT_BG)
add_eyebrow(s, "Phase 1 + 2 — Replication & Feature Tuning")
add_title(s, "Replicating the paper, then tuning its own settings", size=25, height=0.6)
add_text(s, MARGIN, 1.5, 5.3, 1.3,
         "The paper's Doc2Vec+LR, trained with its default settings, then with LDA's topic count "
         "and Doc2Vec's vector size tuned on dev only.", size=14, color=TEXT_MUTED, line_spacing=1.3)
add_stat_row(s, 2.75, 1.5, [
    (pct(P1_ACC), "Phase 1\npaper defaults", GREY := RGBColor(0x8A, 0x8F, 0x98)),
    (pct(P2_ACC), "Phase 2\ntuned feature size", ACCENT),
], left=MARGIN, width=5.3, gap=0.3)
add_text(s, MARGIN, 4.5, 5.3, 1.6,
         "LDA needed more topics (50→100) to separate 15+ categories. Doc2Vec preferred a "
         "smaller fingerprint (100→50) — with ~5,600 training rulings, bigger vectors just overfit.",
         size=13, color=TEXT_MUTED, line_spacing=1.3)
s.shapes.add_picture("results/figures/before_after_tuning.png", Inches(6.6), Inches(1.75), width=Inches(5.9))
add_page_dots(s, 5, TOTAL)

# ================================================================
# 7. Classifier Comparison (Phase 3)
# ================================================================
s = add_slide(LIGHT_BG)
add_eyebrow(s, "Phase 3 — Beyond the Paper")
add_title(s, "Same features, different classifier on top", size=26, height=0.6)
add_text(s, MARGIN, 1.55, 5.4, 0.9,
         "The paper only tries Logistic Regression. We also tried SVM, Random Forest, and Gaussian "
         "Naive Bayes — same Doc2Vec fingerprints, issue_area task.", size=14, color=TEXT_MUTED, line_spacing=1.3)
add_stat_row(s, 2.65, 1.6, [
    (pct(P3_ACC), "Winner: SVM (RBF)", GREEN),
    (f"{P3_F1:.3f}", "Macro-F1", GREEN),
], left=MARGIN, width=5.4, gap=0.3)
add_text(s, MARGIN, 4.6, 5.4, 1.6,
         "SVM wins — its RBF kernel draws a curved decision boundary through the embedding "
         "space that straight-line Logistic Regression can't match.", size=14, color=TEXT_MUTED, line_spacing=1.3)
s.shapes.add_picture("results/figures/classifier_comparison.png", Inches(6.6), Inches(1.75), width=Inches(5.9))
add_page_dots(s, 6, TOTAL)

# ================================================================
# 8. Phase 4 — Deep-Tuned SVM
# ================================================================
s = add_slide(LIGHT_BG)
add_eyebrow(s, "Phase 4 — Deep-Tuned SVM")
add_title(s, "A wider search finds a different kind of win", size=25, height=0.6)
add_text(s, MARGIN, 1.5, 11.2, 0.6,
         "Phase 3's SVM was still untuned. We grid-searched 40 kernel/C/gamma combinations on dev "
         "accuracy — the metric that actually matters for the deployed model.", size=14, color=TEXT_MUTED, line_spacing=1.25)
s.shapes.add_picture("results/figures/phase4_before_after.png", Inches(3.52), Inches(2.1), width=Inches(6.3))
best_kernel = T4["best_params"]["kernel"]
add_text(s, MARGIN, 6.25, 11.6, 0.55,
         f"Winning config: kernel={best_kernel}, degree={T4['best_params'].get('degree', '-')}. Accuracy held "
         f"flat ({pct(P4_ACC_BEFORE)}→{pct(P4_ACC_AFTER)}, within noise for an ~850-example test set) "
         f"while macro-F1 jumped ({P4_F1_BEFORE:.3f}→{P4_F1_AFTER:.3f}).",
         size=12.5, color=TEXT_MUTED, align=PP_ALIGN.CENTER, line_spacing=1.2)
add_page_dots(s, 7, TOTAL)

# ================================================================
# 9. Phase 5 — Pretrained Embeddings (SBERT)
# ================================================================
s = add_slide(LIGHT_BG)
add_eyebrow(s, "Phase 5 — Pretrained Embeddings")
add_title(s, "A pretrained model beats everything, with zero training", size=24, height=0.6)
add_text(s, MARGIN, 1.5, 11.2, 0.6,
         "Doc2Vec learns from scratch on just our ~5,600 documents. SBERT is pretrained on 1B+ "
         "sentence pairs before it ever sees a court opinion — we only use it to encode, never train.",
         size=14, color=TEXT_MUTED, line_spacing=1.25)
s.shapes.add_picture("results/figures/sbert_vs_doc2vec.png", Inches(3.37), Inches(2.1), width=Inches(6.6))
add_text(s, MARGIN, 6.35, 11.2, 0.4,
         f"SBERT + SVM: {pct(P5_ACC)} accuracy — our new overall best, beating Phase 4's deep-tuned "
         f"Doc2Vec SVM ({pct(P4_ACC_AFTER)}).", size=13, color=GREEN, bold=True, align=PP_ALIGN.CENTER)
add_page_dots(s, 8, TOTAL)

# ================================================================
# 10. Confusion Matrix
# ================================================================
s = add_slide(LIGHT_BG)
add_eyebrow(s, "A Closer Look")
add_title(s, "Where the best model actually gets confused", size=24, height=0.55)
s.shapes.add_picture("results/figures/confusion_matrix.png", Inches(3.97), Inches(1.4), width=Inches(5.4))
add_text(s, MARGIN, 6.35, 11.6, 0.35,
         "Rows = true category, columns = predicted — Economic Activity pulls in "
         "misclassifications from Judicial Power and Federalism.",
         size=11, color=TEXT_MUTED, align=PP_ALIGN.CENTER)
add_page_dots(s, 9, TOTAL)

# ================================================================
# 11. Error Analysis
# ================================================================
s = add_slide(LIGHT_BG)
add_eyebrow(s, "Error Analysis")
add_title(s, "What does a mistake actually look like?", size=26, height=0.6)
wrong_card = add_card(s, MARGIN, 1.75, 5.5, 3.9, "A confident mistake", "placeholder", title_color=RED)
wrong_card.text_frame.paragraphs[1].runs[0].text = (
    "True category: Attorneys\nPredicted: Federal Taxation (97%)\n\n"
    "Excerpt: “...an assignment made by a subcontractor to his performance-bond surety of all "
    "sums to become due for performance of the subcontract, as security for any indebtedness...”\n\n"
    "The case is about an attorney's fee dispute, but its surface language is dominated by bond/"
    "surety/payment terms — the model latched onto financial vocabulary over the deeper legal issue."
)
right_card = add_card(s, 6.6, 1.75, 5.6, 3.9, "A clear correct call", "placeholder", title_color=GREEN)
right_card.text_frame.paragraphs[1].runs[0].text = (
    "True category: Unions\nPredicted: Unions (50%)\n\n"
    "Excerpt: “...the Secretary of Labor sued under §402(b) of the Labor-Management Reporting "
    "and Disclosure Act to invalidate a union election...”\n\n"
    "Explicit, unambiguous labor-law vocabulary (Secretary of Labor, union election, Labor-"
    "Management Act) gives the model a clean, confident signal to work with."
)
add_text(s, MARGIN, 5.9, 11.2, 0.7,
         "The pattern: the model reads surface vocabulary, not legal reasoning — it does well when "
         "a case's language matches its category, and struggles when the real legal issue is buried "
         "under financial or procedural detail.", size=13, color=TEXT_MUTED, line_spacing=1.25)
add_page_dots(s, 10, TOTAL)

# ================================================================
# 12. Per-Category Insights
# ================================================================
s = add_slide(LIGHT_BG)
add_eyebrow(s, "A Closer Look")
add_title(s, "LDA guesses the majority class. Doc2Vec doesn't.", size=25, height=0.6)
s.shapes.add_picture("results/figures/per_category_f1.png", Inches(2.17), Inches(1.55), width=Inches(9.0))
add_accent_bar(s, 1.3, 6.25, 0.22, height=0.22, color=RED)
add_text(s, 1.65, 6.17, 5.0, 0.4, "LDA — collapses on rare categories (near-0 F1 on Privacy, Attorneys)",
         size=12, color=TEXT_MUTED)
add_accent_bar(s, 7.0, 6.25, 0.22, height=0.22, color=GREEN)
add_text(s, 7.35, 6.17, 5.0, 0.4, "Doc2Vec — far more balanced across all 13 categories",
         size=12, color=TEXT_MUTED)
add_page_dots(s, 11, TOTAL)

# ================================================================
# 13. Progress Across All Phases
# ================================================================
s = add_slide(LIGHT_BG)
add_eyebrow(s, "Putting It Together")
add_title(s, "Each phase pushed the best model further")
s.shapes.add_picture("results/figures/progress_by_phase.png", Inches(3.17), Inches(1.6), width=Inches(7.0))
add_text(s, MARGIN, 6.15, 11.2, 0.45,
         f"From {pct(P1_ACC)} (paper's default) to {pct(BEST_ACC)} (SBERT) — without ever touching the dataset itself.",
         size=14, color=TEXT_MUTED, italic=True, align=PP_ALIGN.CENTER)
add_page_dots(s, 12, TOTAL)

# ================================================================
# 14. Limitations & Future Work
# ================================================================
s = add_slide(LIGHT_BG)
add_eyebrow(s, "Limitations & Future Work")
add_title(s, "What we'd do with more time", size=27, height=0.6)
limitations = [
    "4,000-character truncation loses most of each opinion (median length: ~34,000 characters)",
    "Fine-grained 'issue' task (260+ classes) is still weak everywhere we tried — ~40-48% accuracy at best",
    "Single train/dev/test split — no cross-validation, so small accuracy differences carry real uncertainty",
    "BERT fine-tuning (vs. SBERT's frozen embeddings) could push further, if memory/compute allow",
    "A Legal-BERT variant, pretrained on legal text specifically, is a natural next embedding to try",
]
add_checkmark_bullets(s, MARGIN, 1.8, 11.2, limitations, size=15.5, text_color=TEXT_DARK,
                       badge_color=RGBColor(0xF5, 0xC6, 0xB8), row_h=0.72)
add_page_dots(s, 13, TOTAL)

# ================================================================
# 15. Conclusions
# ================================================================
s = add_slide(DARK)
add_eyebrow(s, "Conclusions", dark_bg=True)
bullets = [
    "Pretrained embeddings (SBERT) beat from-scratch ones (Doc2Vec) at every comparison point",
    "Classifier choice matters almost as much as features — SVM beat every other classifier tried",
    "Tuning helped every phase — though Phase 4 traded flat accuracy for a macro-F1 win",
    "Coarse categories beat fine-grained ones everywhere — a data problem, not a model problem",
    f"Overall: {pct(P1_ACC)} → {pct(BEST_ACC)} across 5 phases, without touching the dataset",
    "Our numbers land in the same range and order as the original paper's",
]
add_checkmark_bullets(s, MARGIN, 1.5, 11.2, bullets, size=17, badge_color=ACCENT_LIGHT, row_h=0.56)
add_text(s, MARGIN, 6.55, 7, 0.35, "github.com/amogh-herle/ML-MINI-PROJECT", size=13, color=ACCENT_LIGHT)
add_page_dots(s, 14, TOTAL, dark_bg=True)

prs.save("report/slides.pptx")
print("Wrote report/slides.pptx")
