"""
Phase 2: hyperparameter tuning, dev-only selection, test touched once.

Tunes:
  - LDA: num_topics (feature extractor), then LR C per label task
  - Doc2Vec: vector_size (feature extractor), then LR C per label task

Usage:
    python run_all.py     # must run first (downloads/caches the corpus)
    python tune_all.py
"""
import json
import sys
import time

sys.path.insert(0, "src")
from preprocessing import load_corpus, split_records, simple_tokenize
from lda_lr import LDAFeaturizer, train_lda_lr
from doc2vec_lr import Doc2VecFeaturizer, train_doc2vec_lr
from run_all import filter_rare_classes, evaluate_multiclass

SEED = 42


def main():
    t0 = time.time()
    print("Loading + splitting corpus (same seed as run_all.py)...")
    records = load_corpus()
    records = filter_rare_classes(records)
    train, dev, test = split_records(records)

    train_tok = [simple_tokenize(r["text"]) for r in train]
    dev_tok = [simple_tokenize(r["text"]) for r in dev]
    test_tok = [simple_tokenize(r["text"]) for r in test]

    y_area_train = [r["issue_area"] for r in train]
    y_area_dev = [r["issue_area"] for r in dev]
    y_area_test = [r["issue_area"] for r in test]
    y_issue_train = [r["issue"] for r in train]
    y_issue_dev = [r["issue"] for r in dev]
    y_issue_test = [r["issue"] for r in test]

    tuned = {}

    # ---------------- LDA: tune num_topics on issue_area dev accuracy ----------------
    print("Tuning LDA num_topics (selection metric: issue_area dev accuracy)...")
    best_topics, best_acc, best_feat = None, -1, None
    for num_topics in [20, 50, 100]:
        feat = LDAFeaturizer(num_topics=num_topics, passes=5, seed=SEED)
        Xtr = feat.fit_transform(train_tok)
        Xdev = feat.transform(dev_tok)
        clf = train_lda_lr(Xtr, y_area_train, C=1.0)
        acc = evaluate_multiclass(y_area_dev, clf.predict(Xdev))["accuracy"]
        print(f"  num_topics={num_topics} -> dev acc={acc:.4f}")
        if acc > best_acc:
            best_acc, best_topics, best_feat = acc, num_topics, feat
    Xtr_lda = best_feat.transform(train_tok)
    Xdev_lda = best_feat.transform(dev_tok)
    Xtest_lda = best_feat.transform(test_tok)

    for task_name, ytr, ydev, ytest in [
        ("issue_area", y_area_train, y_area_dev, y_area_test),
        ("issue", y_issue_train, y_issue_dev, y_issue_test),
    ]:
        print(f"Tuning LDA+LR C for task={task_name}...")
        best_C, best_f1, best_pred = None, -1, None
        for C in [0.01, 0.1, 1.0, 5.0, 10.0]:
            clf = train_lda_lr(Xtr_lda, ytr, C=C)
            dev_pred = clf.predict(Xdev_lda)
            f1 = evaluate_multiclass(ydev, dev_pred)["macro_f1"]
            if f1 > best_f1:
                best_f1, best_C = f1, C
                best_pred = clf.predict(Xtest_lda)
        key = f"lda_lr_{task_name}"
        tuned[key] = {
            "num_topics": best_topics,
            "best_C": best_C,
            "dev_after_f1": best_f1,
            "test_after": evaluate_multiclass(ytest, best_pred),
        }

    # ---------------- Doc2Vec: tune vector_size on issue_area dev accuracy ----------------
    print("Tuning Doc2Vec vector_size (selection metric: issue_area dev accuracy)...")
    best_size, best_acc, best_feat = None, -1, None
    for vector_size in [50, 100, 200]:
        feat = Doc2VecFeaturizer(vector_size=vector_size, window=8, epochs=20, seed=SEED)
        Xtr = feat.fit_transform(train_tok)
        Xdev = feat.transform(dev_tok)
        clf = train_doc2vec_lr(Xtr, y_area_train, C=1.0)
        acc = evaluate_multiclass(y_area_dev, clf.predict(Xdev))["accuracy"]
        print(f"  vector_size={vector_size} -> dev acc={acc:.4f}")
        if acc > best_acc:
            best_acc, best_size, best_feat = acc, vector_size, feat
    Xtr_d2v = best_feat.transform(train_tok)
    Xdev_d2v = best_feat.transform(dev_tok)
    Xtest_d2v = best_feat.transform(test_tok)

    for task_name, ytr, ydev, ytest in [
        ("issue_area", y_area_train, y_area_dev, y_area_test),
        ("issue", y_issue_train, y_issue_dev, y_issue_test),
    ]:
        print(f"Tuning Doc2Vec+LR C for task={task_name}...")
        best_C, best_f1, best_pred = None, -1, None
        for C in [0.01, 0.1, 1.0, 5.0, 10.0]:
            clf = train_doc2vec_lr(Xtr_d2v, ytr, C=C)
            dev_pred = clf.predict(Xdev_d2v)
            f1 = evaluate_multiclass(ydev, dev_pred)["macro_f1"]
            if f1 > best_f1:
                best_f1, best_C = f1, C
                best_pred = clf.predict(Xtest_d2v)
        key = f"doc2vec_lr_{task_name}"
        tuned[key] = {
            "vector_size": best_size,
            "best_C": best_C,
            "dev_after_f1": best_f1,
            "test_after": evaluate_multiclass(ytest, best_pred),
        }

    with open("results/tuned_results.json", "w") as f:
        json.dump(tuned, f, indent=2, ensure_ascii=False)
    print(json.dumps(tuned, indent=2, ensure_ascii=False))
    print(f"\nWritten to results/tuned_results.json")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
