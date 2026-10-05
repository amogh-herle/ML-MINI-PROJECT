"""Generate a markdown comparison table + bar chart from results/table1.json and
results/table2.json, for the README / write-up."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("results/table1.json") as f:
    table1 = json.load(f)
with open("results/table2.json") as f:
    table2 = json.load(f)

rows = []
for clf_name, toks in table1.items():
    for tok_name, m in toks.items():
        rows.append((clf_name, tok_name, m["train_acc"], m["dev"]["accuracy"],
                      m["dev"]["precision"], m["dev"]["recall"], m["dev"]["f1"]))

md = ["| Classifier | Tokenization | Train Acc | Dev Acc | Dev Precision | Dev Recall | Dev F1 |",
      "|---|---|---|---|---|---|---|"]
for clf_name, tok_name, tr, acc, p, r, f1 in rows:
    md.append(f"| {clf_name} | {tok_name} | {tr:.4f} | {acc:.4f} | {p:.4f} | {r:.4f} | {f1:.4f} |")

md.append("")
md.append("### Table 2 — Best 2 models (by dev F1), test-set performance")
md.append("| Model | Accuracy | Precision | Recall | F1 |")
md.append("|---|---|---|---|---|")
for key, m in table2.items():
    md.append(f"| {key} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |")

try:
    with open("results/tuned_results.json") as f:
        tuned = json.load(f)
    md.append("")
    md.append("### Phase 2 — Before/after tuning (dev-only selection, test touched once)")
    md.append("| Classifier | Tokenization | Best Params | Dev F1 Before | Dev F1 After | Test F1 After |")
    md.append("|---|---|---|---|---|---|")
    for clf_name, r in tuned.items():
        params = ", ".join(f"{k}={v}" for k, v in r["best_params"].items())
        md.append(
            f"| {clf_name} | {r['tokenization']} | {params} | "
            f"{r['dev_before']['f1']:.4f} | {r['dev_after']['f1']:.4f} | {r['test_after']['f1']:.4f} |"
        )
except FileNotFoundError:
    pass

with open("results/comparison_table.md", "w") as f:
    f.write("\n".join(md))
print("Wrote results/comparison_table.md")

# Bar chart: dev F1 by classifier (best tokenization per classifier)
best_per_clf = {}
for clf_name, tok_name, tr, acc, p, r, f1 in rows:
    if clf_name not in best_per_clf or f1 > best_per_clf[clf_name][1]:
        best_per_clf[clf_name] = (tok_name, f1)

names = list(best_per_clf.keys())
f1s = [best_per_clf[n][1] for n in names]
labels = [f"{n}\n({best_per_clf[n][0]})" for n in names]

plt.figure(figsize=(8, 5))
plt.bar(labels, f1s, color="#4C72B0")
plt.ylabel("Dev F1 (best tokenization)")
plt.title("Best Dev F1 per Classifier — Paper Replication")
plt.ylim(0, 1)
for i, v in enumerate(f1s):
    plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
plt.tight_layout()
plt.savefig("results/figures/dev_f1_by_classifier.png", dpi=150)
print("Wrote results/figures/dev_f1_by_classifier.png")

# Before/after tuning chart
try:
    with open("results/tuned_results.json") as f:
        tuned = json.load(f)
    names = list(tuned.keys())
    before = [tuned[n]["dev_before"]["f1"] for n in names]
    after = [tuned[n]["dev_after"]["f1"] for n in names]
    x = range(len(names))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in x], before, width, label="Before tuning (paper defaults)", color="#999999")
    plt.bar([i + width / 2 for i in x], after, width, label="After tuning (dev-selected)", color="#55A868")
    plt.xticks(list(x), names, rotation=20, ha="right")
    plt.ylabel("Dev F1")
    plt.title("Phase 2: Before vs After Hyperparameter Tuning")
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("results/figures/before_after_tuning.png", dpi=150)
    print("Wrote results/figures/before_after_tuning.png")
except FileNotFoundError:
    pass
