"""Shared data-loading and evaluation helpers used by every Phase script
(run_all.py, tune_all.py, compare_classifiers.py, tune_classifiers.py).
Pulled out of run_all.py to stop four scripts from re-implementing the same
"load -> filter -> split -> tokenize" block and importing it from a script.
"""
from collections import Counter

from preprocessing import load_corpus, split_records, simple_tokenize
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

MIN_ISSUE_COUNT = 10  # drop fine-grained issue codes with too few examples to stratify/evaluate
MIN_AREA_COUNT = 2    # drop issue_area classes that can't be stratified


def filter_rare_classes(records, min_issue=MIN_ISSUE_COUNT, min_area=MIN_AREA_COUNT):
    issue_counts = Counter(r["issue"] for r in records)
    area_counts = Counter(r["issue_area"] for r in records)
    return [
        r for r in records
        if issue_counts[r["issue"]] >= min_issue and area_counts[r["issue_area"]] >= min_area
    ]


def load_split_tokenize():
    """Load the corpus, filter rare classes, split 70/15/15, tokenize. Returns
    a dict of tokenized train/dev/test texts plus both label tasks' targets,
    so callers can destructure only what they need."""
    records = filter_rare_classes(load_corpus())
    train, dev, test = split_records(records)
    return {
        "train_tok": [simple_tokenize(r["text"]) for r in train],
        "dev_tok": [simple_tokenize(r["text"]) for r in dev],
        "test_tok": [simple_tokenize(r["text"]) for r in test],
        "issue_area": (
            [r["issue_area"] for r in train], [r["issue_area"] for r in dev], [r["issue_area"] for r in test],
        ),
        "issue": (
            [r["issue"] for r in train], [r["issue"] for r in dev], [r["issue"] for r in test],
        ),
        "counts": (len(train), len(dev), len(test)),
    }


def evaluate_multiclass(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_precision": p, "macro_recall": r, "macro_f1": f1}


def per_class_report(y_true, y_pred, labels):
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        str(label): {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]), "support": int(support[i])}
        for i, label in enumerate(labels)
    }
