"""
Phase 1: replicate the reference paper's experiments —
"Classification of Legal Text" (Iyer, Stanford CS229, 2020).

Paper's Experiment 1 (LDA + LR) and Experiment 2 (Doc2Vec + LR), evaluated on
both label granularities:
  - issue_area: 15 broad categories  (paper's Table 1 "15-Labels" column)
  - issue:      ~260-280 fine issue codes (paper's "279 Labels" column)

Experiment 3 (BERT) is not replicated: the paper's own BERT attempt ran out
of memory and produced no results to compare against (documented in README).

Dataset: real Supreme Court Database corpus via textacy.datasets.SupremeCourt
(~8,400 opinions, 1946-2016) — the same dataset the paper itself used, no
substitute needed this time.

Usage:
    python run_all.py
"""
import json
import sys
import time

sys.path.insert(0, "src")
from pipeline import load_split_tokenize, evaluate_multiclass, per_class_report
from lda_lr import LDAFeaturizer, train_lda_lr
from doc2vec_lr import Doc2VecFeaturizer, train_doc2vec_lr

SEED = 42


def run_experiment(name, featurizer, train_fn, data, results):
    print(f"Training {name} featurizer...")
    Xtr = featurizer.fit_transform(data["train_tok"])
    Xdev = featurizer.transform(data["dev_tok"])
    Xtest = featurizer.transform(data["test_tok"])

    for task in ("issue_area", "issue"):
        ytr, ydev, ytest = data[task]
        print(f"{name}: {task}...")
        clf = train_fn(Xtr, ytr, C=1.0)
        dev_pred, test_pred = clf.predict(Xdev), clf.predict(Xtest)
        results["table1_accuracy"][f"{name}_{task}"] = {
            "dev": evaluate_multiclass(ydev, dev_pred),
            "test": evaluate_multiclass(ytest, test_pred),
        }
        if task == "issue_area":
            results["table2_per_category_issue_area"][name] = per_class_report(
                ytest, test_pred, labels=sorted(set(ytr))
            )


def main():
    t0 = time.time()
    print("Loading Supreme Court corpus (textacy)...")
    data = load_split_tokenize()
    n_train, n_dev, n_test = data["counts"]
    print(f"train={n_train} dev={n_dev} test={n_test}")

    results = {"table1_accuracy": {}, "table2_per_category_issue_area": {}}
    run_experiment("lda_lr", LDAFeaturizer(num_topics=50, passes=5, seed=SEED), train_lda_lr, data, results)
    run_experiment("doc2vec_lr", Doc2VecFeaturizer(vector_size=100, window=8, epochs=20, seed=SEED),
                    train_doc2vec_lr, data, results)

    with open("results/table1.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results["table1_accuracy"], indent=2, ensure_ascii=False))
    print(f"\nWritten to results/table1.json")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
