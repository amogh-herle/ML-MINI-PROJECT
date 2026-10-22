"""Generate a markdown comparison table + bar charts from results/table1.json and
results/tuned_results.json."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Match the deck's color palette so charts drop straight into the slides/write-up
NAVY = "#142340"
ACCENT = "#2A5CAA"
GREEN = "#1E7A3D"
GREY = "#8A8F98"
RED = "#B3261E"
PURPLE = "#6B4FA0"
BG = "#FBFBF8"
TEXT = "#1A1A1A"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "axes.edgecolor": "#D7DAD8",
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
})

with open("results/table1.json") as f:
    r1 = json.load(f)
table1 = r1["table1_accuracy"]
per_cat = r1["table2_per_category_issue_area"]

md = ["### Table 1 — Phase 1 replication: accuracy by model & label granularity",
      "| Model | Label task | Dev Acc | Dev Macro-F1 | Test Acc | Test Macro-F1 |",
      "|---|---|---|---|---|---|"]
for key, r in table1.items():
    if key.endswith("issue_area"):
        model, task = key[: -len("_issue_area")], "issue_area (15-label)"
    else:
        model, task = key[: -len("_issue")], "issue (fine-grained)"
    md.append(
        f"| {model} | {task} | {r['dev']['accuracy']:.4f} | {r['dev']['macro_f1']:.4f} | "
        f"{r['test']['accuracy']:.4f} | {r['test']['macro_f1']:.4f} |"
    )

md.append("")
md.append("### Table 2 — Per-category test performance on issue_area (15 classes)")
for model_name, cats in per_cat.items():
    md.append(f"\n**{model_name}**")
    md.append("| Category | Precision | Recall | F1 | Support |")
    md.append("|---|---|---|---|---|")
    for cat, m in sorted(cats.items(), key=lambda kv: int(kv[0])):
        md.append(f"| {cat} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {m['support']} |")

try:
    with open("results/tuned_results.json") as f:
        tuned = json.load(f)
    md.append("")
    md.append("### Phase 2 — Before/after tuning (dev-selected, test touched once)")
    md.append("| Model+Task | Extra Params | Best C | Dev Macro-F1 (after) | Test Acc (after) | Test Macro-F1 (after) |")
    md.append("|---|---|---|---|---|---|")
    for key, r in tuned.items():
        extra = r.get("num_topics", r.get("vector_size"))
        extra_label = "num_topics" if "num_topics" in r else "vector_size"
        md.append(
            f"| {key} | {extra_label}={extra} | {r['best_C']} | {r['dev_after_f1']:.4f} | "
            f"{r['test_after']['accuracy']:.4f} | {r['test_after']['macro_f1']:.4f} |"
        )
except FileNotFoundError:
    tuned = None

with open("results/comparison_table.md", "w") as f:
    f.write("\n".join(md))
print("Wrote results/comparison_table.md")

# Bar chart: test accuracy by model x task (phase 1)
labels, accs = [], []
for key, r in table1.items():
    labels.append(key)
    accs.append(r["test"]["accuracy"])
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(labels, accs, color=ACCENT, width=0.6)
ax.set_ylabel("Test Accuracy")
ax.set_title("Phase 1: Test Accuracy by Model & Label Task", fontweight="bold", color=NAVY)
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim(0, max(accs) * 1.25)
plt.xticks(rotation=20, ha="right")
for i, v in enumerate(accs):
    plt.text(i, v + 0.012, f"{v:.1%}", ha="center", fontweight="bold", color=NAVY)
plt.tight_layout()
plt.savefig("results/figures/phase1_test_accuracy.png", dpi=150)
print("Wrote results/figures/phase1_test_accuracy.png")

if tuned:
    names = list(tuned.keys())
    before = [table1[n]["test"]["accuracy"] for n in names]
    after = [tuned[n]["test_after"]["accuracy"] for n in names]
    x = range(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar([i - width / 2 for i in x], before, width, label="Phase 1 (default)", color=GREY)
    b2 = ax.bar([i + width / 2 for i in x], after, width, label="Phase 2 (tuned)", color=GREEN)
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(list(x), names, rotation=20, ha="right")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.0%}", ha="center", fontsize=9)
    plt.ylabel("Test Accuracy")
    plt.title("Phase 2: Before vs After Hyperparameter Tuning", fontweight="bold", color=NAVY)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("results/figures/before_after_tuning.png", dpi=150)
    print("Wrote results/figures/before_after_tuning.png")

# ---------------- Phase 3: classifier comparison (SVM / Random Forest / Naive Bayes) ----------------
try:
    with open("results/classifier_comparison.json") as f:
        cc = json.load(f)
    md.append("")
    md.append("### Phase 3 — Classifier comparison (same tuned features, different classifier on top)")
    md.append("Features held fixed at Phase 2's winning settings (LDA num_topics=100, "
               "Doc2Vec vector_size=50). Logistic Regression rows repeated from Phase 2 for reference.")
    md.append("| Classifier | Feature | Label task | Test Acc | Test Macro-F1 |")
    md.append("|---|---|---|---|---|")
    # include Logistic Regression rows for direct comparison
    for key, r in tuned.items():
        feat = "lda" if key.startswith("lda") else "doc2vec"
        task = "issue_area" if key.endswith("issue_area") else "issue"
        md.append(f"| logistic_regression | {feat} | {task} | {r['test_after']['accuracy']:.4f} | {r['test_after']['macro_f1']:.4f} |")
    for key, r in sorted(cc.items(), key=lambda kv: -kv[1]["test"]["accuracy"]):
        md.append(f"| {r['classifier']} | {r['feature']} | {r['label_task']} | {r['test']['accuracy']:.4f} | {r['test']['macro_f1']:.4f} |")

    with open("results/comparison_table.md", "w") as f:
        f.write("\n".join(md))
    print("Wrote results/comparison_table.md (with Phase 3)")

    # chart: best classifier per (feature, task) combo, grouped
    combos = sorted(set((r["feature"], r["label_task"]) for r in cc.values()))
    clf_names = ["logistic_regression", "svm", "random_forest", "naive_bayes"]
    colors_map = {"logistic_regression": ACCENT, "svm": GREEN, "random_forest": PURPLE, "naive_bayes": GREY}
    x = range(len(combos))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, clf_name in enumerate(clf_names):
        vals = []
        for feat, task in combos:
            if clf_name == "logistic_regression":
                tkey = f"{feat}_lr_{task}"  # tuned.json naming: lda_lr_issue_area / doc2vec_lr_issue
                vals.append(tuned[tkey]["test_after"]["accuracy"])
            else:
                key = f"{clf_name}_{feat}_{task}"
                vals.append(cc[key]["test"]["accuracy"])
        offsets = [xi + (i - 1.5) * width for xi in x]
        bars = ax.bar(offsets, vals, width, label=clf_name.replace("_", " ").title(), color=colors_map[clf_name])
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(list(x), [f"{f}\n{t}" for f, t in combos])
    plt.ylabel("Test Accuracy")
    plt.title("Phase 3: Classifier Comparison (fixed tuned features)", fontweight="bold", color=NAVY)
    plt.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    plt.tight_layout()
    plt.savefig("results/figures/classifier_comparison.png", dpi=150)
    print("Wrote results/figures/classifier_comparison.png")
except FileNotFoundError:
    pass

# ---------------- Per-category F1: LDA vs Doc2Vec (issue_area, 13 classes) ----------------
AREA_NAMES = {
    "1": "Criminal Proc.", "2": "Civil Rights", "3": "1st Amendment", "4": "Due Process",
    "5": "Privacy", "6": "Attorneys", "7": "Unions", "8": "Economic Act.",
    "9": "Judicial Power", "10": "Federalism", "11": "Interstate Rel.", "12": "Fed. Taxation",
    "13": "Misc.",
}
cats_sorted = sorted(per_cat["lda_lr"].keys(), key=int)
lda_f1 = [per_cat["lda_lr"][c]["f1"] for c in cats_sorted]
d2v_f1 = [per_cat["doc2vec_lr"][c]["f1"] for c in cats_sorted]
labels_cat = [AREA_NAMES.get(c, c) for c in cats_sorted]
x = range(len(cats_sorted))
width = 0.38
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.bar([i - width / 2 for i in x], lda_f1, width, label="LDA + LR", color=RED)
ax.bar([i + width / 2 for i in x], d2v_f1, width, label="Doc2Vec + LR", color=GREEN)
ax.spines[["top", "right"]].set_visible(False)
plt.xticks(list(x), labels_cat, rotation=35, ha="right")
plt.ylabel("Test F1")
plt.title("Per-Category F1: LDA vs Doc2Vec (issue_area)", fontweight="bold", color=NAVY)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("results/figures/per_category_f1.png", dpi=150)
print("Wrote results/figures/per_category_f1.png")

# ---------------- Progress across phases: Doc2Vec + issue_area test accuracy ----------------
if tuned:
    try:
        with open("results/classifier_comparison.json") as f:
            cc = json.load(f)
        svm_doc2vec_area = next(
            r for r in cc.values()
            if r["classifier"] == "svm" and r["feature"] == "doc2vec" and r["label_task"] == "issue_area"
        )
        phase_labels = ["Phase 1\n(default, LR)", "Phase 2\n(tuned, LR)", "Phase 3\n(tuned, SVM)"]
        phase_vals = [
            table1["doc2vec_lr_issue_area"]["test"]["accuracy"],
            tuned["doc2vec_lr_issue_area"]["test_after"]["accuracy"],
            svm_doc2vec_area["test"]["accuracy"],
        ]
        fig, ax = plt.subplots(figsize=(8, 5.5))
        colors_seq = [GREY, ACCENT, GREEN]
        bars = ax.bar(phase_labels, phase_vals, color=colors_seq, width=0.55)
        ax.plot(range(3), phase_vals, color=NAVY, linewidth=2, marker="o", markersize=8, zorder=5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, max(phase_vals) * 1.3)
        plt.ylabel("Test Accuracy")
        plt.title("Best Model's Accuracy, Phase by Phase\n(Doc2Vec + issue_area)", fontweight="bold", color=NAVY)
        for i, v in enumerate(phase_vals):
            plt.text(i, v + 0.025, f"{v:.1%}", ha="center", fontweight="bold", fontsize=14, color=NAVY)
        plt.tight_layout()
        plt.savefig("results/figures/progress_by_phase.png", dpi=150)
        print("Wrote results/figures/progress_by_phase.png")
    except (FileNotFoundError, StopIteration):
        pass
