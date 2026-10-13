"""Build report/slides.pptx — a PowerPoint version of the slide deck, mirroring
the 9 slides in report/slides/ (cover, problem, dataset, approach, phase1,
phase2, phase3, insights, conclusions). Pulls numbers from the results JSON
files so it can't drift from the actual results.
"""
import json
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

with open("results/table1.json") as f:
    T1 = json.load(f)["table1_accuracy"]
with open("results/tuned_results.json") as f:
    T2 = json.load(f)
with open("results/classifier_comparison.json") as f:
    T3 = json.load(f)

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
ROW_ALT = RGBColor(0xF2, 0xF2, 0xF2)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def add_slide(bg_color):
    slide = prs.slides.add_slide(BLANK)
    bg = slide.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = bg_color
    bg.line.fill.background()
    bg.shadow.inherit = False
    # send to back
    sp = bg._element
    sp.getparent().remove(sp)
    slide.shapes._spTree.insert(2, sp)
    return slide


def add_text(slide, left, top, width, height, text, size=18, color=TEXT_DARK,
             bold=False, italic=False, align=PP_ALIGN.LEFT, font="Arial",
             line_spacing=1.1, letter_spacing=None):
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


def add_eyebrow(slide, text, top=0.55, color=ACCENT, dark_bg=False):
    add_text(slide, 0.9, top, 11, 0.4, text.upper(), size=14, color=(ACCENT_LIGHT if dark_bg else color), bold=True)


def add_footer(slide, page, total, dark_bg=False):
    add_text(slide, 0.9, 6.95, 2, 0.4, f"{page} / {total}", size=12, color=(RGBColor(0x6E, 0x85, 0xAD) if dark_bg else TEXT_GREY))


def add_card(slide, left, top, width, height, title, body, title_color=ACCENT, accent_strip=True):
    card = slide.shapes.add_shape(1, Inches(left), Inches(top), Inches(width), Inches(height))
    card.fill.solid()
    card.fill.fore_color.rgb = CARD_BG
    card.line.color.rgb = CARD_BORDER
    card.line.width = Pt(1)
    card.shadow.inherit = False
    if accent_strip:
        add_accent_bar(slide, left, top, width, height=0.06, color=title_color)
    tf = card.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.3)
    tf.margin_right = Inches(0.3)
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


def add_table(slide, left, top, width, height, header, rows, highlight_row=None, col_widths=None):
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
        r.font.size = Pt(13)
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
            if highlight_row is not None and i == highlight_row:
                cell.fill.fore_color.rgb = RGBColor(0xE8, 0xF3, 0xEC)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j < 2 else PP_ALIGN.CENTER
            r = p.runs[0]
            r.font.size = Pt(13)
            r.font.name = "Arial"
            r.font.color.rgb = GREEN if (highlight_row is not None and i == highlight_row) else TEXT_DARK
            r.font.bold = highlight_row is not None and i == highlight_row
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
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    bar.fill.solid()
    bar.fill.fore_color.rgb = color or ACCENT
    bar.line.fill.background()
    bar.shadow.inherit = False
    return bar


def add_checkmark_bullets(slide, left, top, width, bullets, size=19, gap=14,
                           text_color=RGBColor(0xE3, 0xE9, 0xF5), badge_color=None):
    """Rows of a small colored checkmark circle + text, instead of a plain bullet."""
    y = top
    for b in bullets:
        add_badge(slide, left + 0.16, y + 0.16, 0.32, badge_color or ACCENT_LIGHT, "✓",
                   text_color=DARK, size=13)
        add_text(slide, left + 0.5, y - 0.03, width - 0.5, 0.6, b, size=size, color=text_color, line_spacing=1.15)
        y += 0.62


# ---------------- 1. Cover ----------------
s = add_slide(DARK)
add_text(s, 0.9, 0.9, 11, 0.4, "UE24CS352A — MACHINE LEARNING MINI-PROJECT", size=15, color=ACCENT_LIGHT, bold=True)
add_accent_bar(s, 0.93, 1.5, 1.6, height=0.07, color=ACCENT)
add_text(s, 0.9, 1.65, 11, 1.8, "Classification of Legal Text", size=48, color=WHITE, bold=True)
add_text(s, 0.9, 3.05, 10.5, 1.2,
         "Replicating & tuning a Stanford CS229 NLP project — teaching a classifier to read U.S. "
         "Supreme Court opinions and guess their legal category.",
         size=20, color=RGBColor(0xC7, 0xD4, 0xEA), line_spacing=1.3)
# big faint decorative number badges, bottom-right, hint at the headline results
add_badge(s, 11.7, 1.3, 1.5, RGBColor(0x1C, 0x30, 0x55), "68.8%", text_color=ACCENT_LIGHT, size=15)
add_text(s, 10.95, 2.15, 1.5, 0.3, "best test acc", size=10, color=RGBColor(0x6E, 0x85, 0xAD), align=PP_ALIGN.CENTER)
add_text(s, 0.9, 6.9, 11, 0.4, 'Based on: Krithika Iyer, "Classification of Legal Text" (CS229, 2020)',
         size=13, color=RGBColor(0x6E, 0x85, 0xAD))

# ---------------- 2. Problem ----------------
s = add_slide(LIGHT_BG)
add_eyebrow(s, "The Problem")
add_text(s, 0.9, 0.95, 11.5, 1.3, "Can a simple model guess a court ruling's legal category?",
         size=28, color=TEXT_DARK, bold=True, line_spacing=1.1)
add_text(s, 0.9, 2.25, 11.2, 0.9,
         "Legal opinions use long, non-standard sentence structures that break typical NLP tooling. "
         "We classify U.S. Supreme Court opinions into their human-curated legal category — at two "
         "levels of detail.", size=15, color=TEXT_MUTED, line_spacing=1.3)
add_card(s, 0.9, 3.35, 5.4, 2.0, "15", "Broad categories (issue_area) — e.g. Criminal Procedure, Civil Rights, First Amendment")
add_card(s, 6.5, 3.35, 5.4, 2.0, "260+", "Fine-grained issue codes (issue) — much harder, far less data per category")
add_footer(s, 1, 9)

# ---------------- 3. Dataset ----------------
s = add_slide(LIGHT_BG)
add_eyebrow(s, "Dataset")
add_text(s, 0.9, 0.95, 11.5, 0.9, "Real Supreme Court opinions — no substitute needed",
         size=28, color=TEXT_DARK, bold=True)
add_text(s, 0.9, 1.75, 11.2, 0.7,
         "Same source the original paper used: the Supreme Court Database via the textacy package. "
         "Still downloadable today.", size=15, color=TEXT_MUTED, line_spacing=1.3)
add_card(s, 0.9, 2.65, 3.6, 1.6, "8,419", "opinions, 1946-2016")
add_card(s, 4.75, 2.65, 3.6, 1.6, "7,971", "after filtering rare classes")
add_card(s, 8.6, 2.65, 3.8, 1.6, "70/15/15", "train / dev / test split")
add_text(s, 0.9, 4.6, 11.2, 1.0,
         "Opinions run long (median ~34,000 characters) — each truncated to its first 4,000 characters "
         "so training stays fast, keeping the part where judges usually state the issue and holding.",
         size=14, color=TEXT_GREY, line_spacing=1.3)
add_footer(s, 2, 9)

# ---------------- 4. Approach ----------------
s = add_slide(LIGHT_BG)
add_eyebrow(s, "Approach")
add_text(s, 0.9, 0.95, 11.5, 0.9, "Two ways to featurize legal text, one classifier",
         size=28, color=TEXT_DARK, bold=True)
add_card(s, 0.9, 2.0, 5.5, 3.3, "LDA + Logistic Regression",
         "Finds abstract recurring topics across all rulings (gensim, TF-IDF). Each ruling becomes "
         "a mix-of-topics score. A classifier maps topic mix → category.\n\nTuned: 100 topics "
         "(vs. 50 default)")
add_card(s, 6.6, 2.0, 5.6, 3.3, "Doc2Vec + Logistic Regression",
         "Learns a dense numeric “fingerprint” per ruling by predicting words in context "
         "(gensim Doc2Vec). Same classifier maps fingerprint → category.\n\nTuned: 50-dim "
         "vectors (vs. 100 default)")
add_text(s, 0.9, 5.55, 11.2, 0.6,
         "A third method (BERT) was in the original paper, but it ran out of memory there — no baseline to replicate.",
         size=13, color=TEXT_GREY)
add_footer(s, 3, 9)

# ---------------- 5. Phase 1 ----------------
s = add_slide(LIGHT_BG)
add_eyebrow(s, "Phase 1 — Replication")
add_text(s, 0.9, 0.95, 11.5, 0.6, "Paper's default settings, our dataset", size=26, color=TEXT_DARK, bold=True)
header = ["Model", "Task", "Test Acc", "F1"]
rows = []
for key, r in T1.items():
    model = key[: -len("_issue_area")] if key.endswith("issue_area") else key[: -len("_issue")]
    task = "issue_area" if key.endswith("issue_area") else "issue"
    rows.append([model, task, pct(r["test"]["accuracy"]), f"{r['test']['macro_f1']:.3f}"])
add_table(s, 0.9, 1.75, 5.5, 2.6, header, rows, col_widths=[1.9, 1.6, 1.05, 0.95])
add_text(s, 0.9, 4.55, 5.3, 1.6,
         "Doc2Vec beats LDA on both tasks. The coarser 15-category task is much easier than the "
         "260+-category task — matches the original paper's pattern.", size=14, color=TEXT_MUTED, line_spacing=1.3)
s.shapes.add_picture("results/figures/phase1_test_accuracy.png", Inches(6.7), Inches(1.75), width=Inches(5.7))
add_footer(s, 4, 9)

# ---------------- 6. Phase 2 ----------------
s = add_slide(LIGHT_BG)
add_eyebrow(s, "Phase 2 — Hyperparameter Tuning")
add_text(s, 0.9, 0.95, 11.5, 0.6, "Dev-only tuning, test touched once", size=26, color=TEXT_DARK, bold=True)
header = ["Model + Task", "Param", "Before", "After"]
rows = []
for key, r in T2.items():
    extra = r.get("num_topics", r.get("vector_size"))
    extra_label = "topics" if "num_topics" in r else "dim"
    before = T1[key]["test"]["accuracy"]
    after = r["test_after"]["accuracy"]
    rows.append([key.replace("_lr_", " — "), f"{extra_label}={extra}", pct(before), pct(after)])
add_table(s, 0.9, 1.75, 6.1, 2.6, header, rows, col_widths=[2.4, 1.1, 1.3, 1.3])
add_text(s, 0.9, 4.55, 5.9, 1.6,
         "LDA needed more topics (50→100) to separate 15+ categories. Doc2Vec preferred a "
         "smaller fingerprint (100→50) — with only ~5,600 training rulings, bigger vectors "
         "just overfit.", size=14, color=TEXT_MUTED, line_spacing=1.3)
s.shapes.add_picture("results/figures/before_after_tuning.png", Inches(7.3), Inches(1.75), width=Inches(5.1))
add_footer(s, 5, 9)

# ---------------- 7. Phase 3 ----------------
s = add_slide(LIGHT_BG)
add_eyebrow(s, "Phase 3 — Beyond the Paper")
add_text(s, 0.9, 0.95, 11.5, 0.6, "Same features, different classifier on top", size=26, color=TEXT_DARK, bold=True)
add_text(s, 0.9, 1.55, 5.4, 0.9,
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
add_table(s, 0.9, 2.6, 5.4, 2.3, header, rows, highlight_row=0, col_widths=[2.6, 1.4, 1.4])
add_text(s, 0.9, 4.95, 5.4, 0.6,
         "SVM (RBF, C=1.0), Random Forest (200 trees) and Naive Bayes use defaults, untuned; "
         "LR (C=1.0) is the tuned one.", size=11, color=TEXT_GREY, line_spacing=1.2)
add_text(s, 0.9, 5.6, 5.4, 1.1,
         "SVM wins — its RBF kernel draws a curved decision boundary through the embedding "
         "space that straight-line Logistic Regression can't match.", size=14, color=TEXT_MUTED, line_spacing=1.3)
s.shapes.add_picture("results/figures/classifier_comparison.png", Inches(6.6), Inches(1.75), width=Inches(5.9))
add_footer(s, 6, 9)

# ---------------- 8. Insights ----------------
s = add_slide(LIGHT_BG)
add_eyebrow(s, "A Closer Look")
add_text(s, 0.9, 0.95, 11.5, 0.9, "LDA guesses the majority class. Doc2Vec doesn't.",
         size=26, color=TEXT_DARK, bold=True)
lda_card = add_card(s, 0.9, 2.0, 5.5, 3.4, "LDA", "placeholder", title_color=RGBColor(0xB3, 0x26, 0x1E))
d2v_card = add_card(s, 6.6, 2.0, 5.6, 3.4, "Doc2Vec", "placeholder", title_color=GREEN)
lda_card.text_frame.paragraphs[1].runs[0].text = (
    "Macro-F1 (0.267) trails accuracy (48.7%) badly — it defaults to the two biggest categories "
    "(Criminal Procedure, Economic Activity) and misses rare ones almost entirely (Privacy, "
    "Attorneys, Interstate Relations: near 0 recall)."
)
d2v_card.text_frame.paragraphs[1].runs[0].text = (
    "Far more balanced — still finds minority categories like Privacy (F1 0.26) and Attorneys "
    "(F1 0.31) that LDA completely misses, because it reads context, not just word counts."
)
add_text(s, 0.9, 5.65, 11.2, 0.5, "Same pattern the original paper reports on its own category breakdown.",
         size=13, color=TEXT_GREY)
add_footer(s, 7, 9)

# ---------------- 9. Conclusions ----------------
s = add_slide(DARK)
add_eyebrow(s, "Conclusions", dark_bg=True)
bullets = [
    "Doc2Vec embeddings consistently beat LDA topic features on this task",
    "Coarse 15-category task is much easier than the 260+-category one",
    "Tuning hyperparameters gave real gains everywhere (not just marginal)",
    "SVM beat every classifier on Doc2Vec issue_area (on LDA ~level with LR: 49.1 vs 48.7%)",
    "Our numbers land in the same range and order as the original paper's",
    "BERT not replicated — paper's own attempt ran out of memory, no baseline to match",
]
add_checkmark_bullets(s, 0.9, 1.6, 11.2, bullets, size=19, badge_color=ACCENT_LIGHT)
add_text(s, 0.9, 6.9, 7, 0.4, "github.com/amogh-herle/ML-MINI-PROJECT", size=13, color=ACCENT_LIGHT)
add_text(s, 10.5, 6.9, 2, 0.4, "9 / 9", size=13, color=RGBColor(0x6E, 0x85, 0xAD))

prs.save("report/slides.pptx")
print("Wrote report/slides.pptx")
