"""Generate a markdown comparison table + bar charts from results/table1.json and
results/tuned_results.json."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
plt.figure(figsize=(9, 5))
plt.bar(labels, accs, color="#4C72B0")
plt.ylabel("Test Accuracy")
plt.title("Phase 1: Test Accuracy by Model & Label Task")
plt.xticks(rotation=30, ha="right")
for i, v in enumerate(accs):
    plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
plt.tight_layout()
plt.savefig("results/figures/phase1_test_accuracy.png", dpi=150)
print("Wrote results/figures/phase1_test_accuracy.png")

if tuned:
    names = list(tuned.keys())
    before = [table1[n]["test"]["accuracy"] for n in names]
    after = [tuned[n]["test_after"]["accuracy"] for n in names]
    x = range(len(names))
    width = 0.35
    plt.figure(figsize=(9, 5))
    plt.bar([i - width / 2 for i in x], before, width, label="Phase 1 (default hyperparams)", color="#999999")
    plt.bar([i + width / 2 for i in x], after, width, label="Phase 2 (tuned)", color="#55A868")
    plt.xticks(list(x), names, rotation=30, ha="right")
    plt.ylabel("Test Accuracy")
    plt.title("Phase 2: Before vs After Hyperparameter Tuning")
    plt.legend()
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
    colors_map = {"logistic_regression": "#4C72B0", "svm": "#C44E52", "random_forest": "#55A868", "naive_bayes": "#8172B2"}
    x = range(len(combos))
    width = 0.2
    plt.figure(figsize=(10, 5.5))
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
        plt.bar(offsets, vals, width, label=clf_name, color=colors_map[clf_name])
    plt.xticks(list(x), [f"{f}\n{t}" for f, t in combos])
    plt.ylabel("Test Accuracy")
    plt.title("Phase 3: Classifier Comparison (fixed tuned features)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/classifier_comparison.png", dpi=150)
    print("Wrote results/figures/classifier_comparison.png")
except FileNotFoundError:
    pass
