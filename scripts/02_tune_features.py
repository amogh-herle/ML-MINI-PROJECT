"""
Phase 2: hyperparameter tuning, dev-only selection, test touched once.

Tunes:
  - LDA: num_topics (feature extractor), then LR C per label task
  - Doc2Vec: vector_size (feature extractor), then LR C per label task

Usage:
    python scripts/01_replicate.py       # must run first (downloads/caches the corpus)
    python scripts/02_tune_features.py
"""
import json
import sys
import time

sys.path.insert(0, "src")
from data.pipeline import load_split_tokenize, evaluate_multiclass
from models.lda_lr import LDAFeaturizer, train_lda_lr
from models.doc2vec_lr import Doc2VecFeaturizer, train_doc2vec_lr

SEED = 42
C_GRID = [0.01, 0.1, 1.0, 5.0, 10.0]


def best_by_dev_accuracy(label, sizes, build_featurizer, train_lr, train_tok, dev_tok, y_area_train, y_area_dev):
    """Sweep a feature extractor's main hyperparameter, picking the value that
    gives the best issue_area dev accuracy with a default-C classifier."""
    best_size, best_acc, best_feat = None, -1, None
    for size in sizes:
        feat = build_featurizer(size)
        Xtr, Xdev = feat.fit_transform(train_tok), feat.transform(dev_tok)
        acc = evaluate_multiclass(y_area_dev, train_lr(Xtr, y_area_train, C=1.0).predict(Xdev))["accuracy"]
        print(f"  {label}={size} -> dev acc={acc:.4f}")
        if acc > best_acc:
            best_acc, best_size, best_feat = acc, size, feat
    return best_size, best_feat


def tune_C(train_fn, Xtr, Xdev, Xtest, data):
    """For each label task, grid-search C on dev macro-F1, touch test once."""
    out = {}
    for task in ("issue_area", "issue"):
        ytr, ydev, ytest = data[task]
        print(f"  tuning C for task={task}...")
        best_C, best_f1, best_clf = None, -1, None
        for C in C_GRID:
            clf = train_fn(Xtr, ytr, C=C)
            f1 = evaluate_multiclass(ydev, clf.predict(Xdev))["macro_f1"]
            if f1 > best_f1:
                best_f1, best_C, best_clf = f1, C, clf
        out[task] = {
            "best_C": best_C, "dev_after_f1": best_f1,
            "test_after": evaluate_multiclass(ytest, best_clf.predict(Xtest)),
        }
    return out


def main():
    t0 = time.time()
    print("Loading + splitting corpus (same seed as run_all.py)...")
    data = load_split_tokenize()
    y_area_train, y_area_dev, _ = data["issue_area"]
    tuned = {}

    print("Tuning LDA num_topics (selection metric: issue_area dev accuracy)...")
    num_topics, lda_feat = best_by_dev_accuracy(
        "num_topics", [20, 50, 100], lambda n: LDAFeaturizer(num_topics=n, passes=5, seed=SEED), train_lda_lr,
        data["train_tok"], data["dev_tok"], y_area_train, y_area_dev,
    )
    Xtr, Xdev, Xtest = (lda_feat.transform(data[k]) for k in ("train_tok", "dev_tok", "test_tok"))
    for task, r in tune_C(train_lda_lr, Xtr, Xdev, Xtest, data).items():
        tuned[f"lda_lr_{task}"] = {"num_topics": num_topics, **r}

    print("Tuning Doc2Vec vector_size (selection metric: issue_area dev accuracy)...")
    vector_size, d2v_feat = best_by_dev_accuracy(
        "vector_size", [50, 100, 200], lambda n: Doc2VecFeaturizer(vector_size=n, window=8, epochs=20, seed=SEED),
        train_doc2vec_lr, data["train_tok"], data["dev_tok"], y_area_train, y_area_dev,
    )
    Xtr, Xdev, Xtest = (d2v_feat.transform(data[k]) for k in ("train_tok", "dev_tok", "test_tok"))
    for task, r in tune_C(train_doc2vec_lr, Xtr, Xdev, Xtest, data).items():
        tuned[f"doc2vec_lr_{task}"] = {"vector_size": vector_size, **r}

    with open("results/tuned_results.json", "w") as f:
        json.dump(tuned, f, indent=2, ensure_ascii=False)
    print(json.dumps(tuned, indent=2, ensure_ascii=False))
    print(f"\nWritten to results/tuned_results.json")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
