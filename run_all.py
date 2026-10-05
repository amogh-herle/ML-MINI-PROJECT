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
from collections import Counter

sys.path.insert(0, "src")
from preprocessing import load_corpus, split_records, simple_tokenize
from lda_lr import LDAFeaturizer, train_lda_lr
from doc2vec_lr import Doc2VecFeaturizer, train_doc2vec_lr
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

SEED = 42
MIN_ISSUE_COUNT = 10   # drop fine-grained issue codes with too few examples to stratify/evaluate
MIN_AREA_COUNT = 2      # drop issue_area classes that can't be stratified


def filter_rare_classes(records):
    issue_counts = Counter(r["issue"] for r in records)
    area_counts = Counter(r["issue_area"] for r in records)
    return [
        r for r in records
        if issue_counts[r["issue"]] >= MIN_ISSUE_COUNT and area_counts[r["issue_area"]] >= MIN_AREA_COUNT
    ]


def evaluate_multiclass(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_precision": p, "macro_recall": r, "macro_f1": f1}


def per_class_report(y_true, y_pred, labels):
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        str(label): {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]), "support": int(support[i])}
        for i, label in enumerate(labels)
    }


def main():
    t0 = time.time()
    print("Loading Supreme Court corpus (textacy)...")
    records = load_corpus()
    records = filter_rare_classes(records)
    print(f"{len(records)} records after filtering rare classes")

    train, dev, test = split_records(records)
    print(f"train={len(train)} dev={len(dev)} test={len(test)}")

    print("Tokenizing...")
    train_tok = [simple_tokenize(r["text"]) for r in train]
    dev_tok = [simple_tokenize(r["text"]) for r in dev]
    test_tok = [simple_tokenize(r["text"]) for r in test]

    y_area_train = [r["issue_area"] for r in train]
    y_area_dev = [r["issue_area"] for r in dev]
    y_area_test = [r["issue_area"] for r in test]
    y_issue_train = [r["issue"] for r in train]
    y_issue_dev = [r["issue"] for r in dev]
    y_issue_test = [r["issue"] for r in test]

    results = {"table1_accuracy": {}, "table2_per_category_issue_area": {}}

    # ---------------- Experiment 1: LDA + LR ----------------
    print("Training LDA featurizer (num_topics=50)...")
    lda_feat = LDAFeaturizer(num_topics=50, passes=5, seed=SEED)
    Xtr_lda = lda_feat.fit_transform(train_tok)
    Xdev_lda = lda_feat.transform(dev_tok)
    Xtest_lda = lda_feat.transform(test_tok)

    print("LDA + LR: issue_area (15-label)...")
    clf = train_lda_lr(Xtr_lda, y_area_train, C=1.0)
    dev_pred = clf.predict(Xdev_lda)
    test_pred = clf.predict(Xtest_lda)
    results["table1_accuracy"]["lda_lr_issue_area"] = {
        "dev": evaluate_multiclass(y_area_dev, dev_pred),
        "test": evaluate_multiclass(y_area_test, test_pred),
    }
    results["table2_per_category_issue_area"]["lda_lr"] = per_class_report(
        y_area_test, test_pred, labels=sorted(set(y_area_train))
    )

    print("LDA + LR: issue (fine-grained)...")
    clf = train_lda_lr(Xtr_lda, y_issue_train, C=1.0)
    dev_pred = clf.predict(Xdev_lda)
    test_pred = clf.predict(Xtest_lda)
    results["table1_accuracy"]["lda_lr_issue"] = {
        "dev": evaluate_multiclass(y_issue_dev, dev_pred),
        "test": evaluate_multiclass(y_issue_test, test_pred),
    }

    # ---------------- Experiment 2: Doc2Vec + LR ----------------
    print("Training Doc2Vec featurizer (vector_size=100)...")
    d2v_feat = Doc2VecFeaturizer(vector_size=100, window=8, epochs=20, seed=SEED)
    Xtr_d2v = d2v_feat.fit_transform(train_tok)
    Xdev_d2v = d2v_feat.transform(dev_tok)
    Xtest_d2v = d2v_feat.transform(test_tok)

    print("Doc2Vec + LR: issue_area (15-label)...")
    clf = train_doc2vec_lr(Xtr_d2v, y_area_train, C=1.0)
    dev_pred = clf.predict(Xdev_d2v)
    test_pred = clf.predict(Xtest_d2v)
    results["table1_accuracy"]["doc2vec_lr_issue_area"] = {
        "dev": evaluate_multiclass(y_area_dev, dev_pred),
        "test": evaluate_multiclass(y_area_test, test_pred),
    }
    results["table2_per_category_issue_area"]["doc2vec_lr"] = per_class_report(
        y_area_test, test_pred, labels=sorted(set(y_area_train))
    )

    print("Doc2Vec + LR: issue (fine-grained)...")
    clf = train_doc2vec_lr(Xtr_d2v, y_issue_train, C=1.0)
    dev_pred = clf.predict(Xdev_d2v)
    test_pred = clf.predict(Xtest_d2v)
    results["table1_accuracy"]["doc2vec_lr_issue"] = {
        "dev": evaluate_multiclass(y_issue_dev, dev_pred),
        "test": evaluate_multiclass(y_issue_test, test_pred),
    }

    with open("results/table1.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results["table1_accuracy"], indent=2, ensure_ascii=False))
    print(f"\nWritten to results/table1.json")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
