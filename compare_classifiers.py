"""
Phase 3: classifier comparison.

Beyond the paper's own Logistic Regression, compare SVM, Random Forest, and
Naive Bayes on the SAME feature vectors (LDA topic vectors at num_topics=100,
Doc2Vec embeddings at vector_size=50 -- the Phase 2 tuned-winning settings),
for both label tasks (issue_area 15-label, issue fine-grained). This isolates
the effect of classifier choice from the effect of feature extraction, which
Phase 1/2 already tuned.

Usage:
    python run_all.py        # must run first (downloads/caches corpus)
    python tune_all.py       # must run first (confirms best feature settings)
    python compare_classifiers.py
"""
import json
import sys
import time

sys.path.insert(0, "src")
from preprocessing import load_corpus, split_records, simple_tokenize
from lda_lr import LDAFeaturizer
from doc2vec_lr import Doc2VecFeaturizer
from classifiers import build_svm, build_random_forest, build_naive_bayes, train_predict
from run_all import filter_rare_classes, evaluate_multiclass

SEED = 42
LDA_NUM_TOPICS = 100     # Phase 2 winner
DOC2VEC_VECTOR_SIZE = 50  # Phase 2 winner


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

    print(f"Building LDA features (num_topics={LDA_NUM_TOPICS}, tuned winner)...")
    lda_feat = LDAFeaturizer(num_topics=LDA_NUM_TOPICS, passes=5, seed=SEED)
    Xtr_lda = lda_feat.fit_transform(train_tok)
    Xdev_lda = lda_feat.transform(dev_tok)
    Xtest_lda = lda_feat.transform(test_tok)

    print(f"Building Doc2Vec features (vector_size={DOC2VEC_VECTOR_SIZE}, tuned winner)...")
    d2v_feat = Doc2VecFeaturizer(vector_size=DOC2VEC_VECTOR_SIZE, window=8, epochs=20, seed=SEED)
    Xtr_d2v = d2v_feat.fit_transform(train_tok)
    Xdev_d2v = d2v_feat.transform(dev_tok)
    Xtest_d2v = d2v_feat.transform(test_tok)

    feature_sets = {
        "lda": (Xtr_lda, Xdev_lda, Xtest_lda),
        "doc2vec": (Xtr_d2v, Xdev_d2v, Xtest_d2v),
    }
    label_tasks = {
        "issue_area": (y_area_train, y_area_dev, y_area_test),
        "issue": (y_issue_train, y_issue_dev, y_issue_test),
    }
    classifiers = {
        "svm": lambda: build_svm(C=1.0, kernel="rbf"),
        "random_forest": lambda: build_random_forest(n_estimators=200, seed=SEED),
        "naive_bayes": lambda: build_naive_bayes(),
    }

    results = {}
    for feat_name, (Xtr, Xdev, Xtest) in feature_sets.items():
        for task_name, (ytr, ydev, ytest) in label_tasks.items():
            for clf_name, clf_builder in classifiers.items():
                key = f"{clf_name}_{feat_name}_{task_name}"
                print(f"Training {key}...")
                clf = clf_builder()
                dev_pred, test_pred = train_predict(clf, Xtr, ytr, Xdev, Xtest)
                results[key] = {
                    "feature": feat_name,
                    "label_task": task_name,
                    "classifier": clf_name,
                    "dev": evaluate_multiclass(ydev, dev_pred),
                    "test": evaluate_multiclass(ytest, test_pred),
                }

    with open("results/classifier_comparison.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n--- Summary (test accuracy) ---")
    for key, r in sorted(results.items(), key=lambda kv: -kv[1]["test"]["accuracy"]):
        print(f"  {key:45s} acc={r['test']['accuracy']:.4f}  macro_f1={r['test']['macro_f1']:.4f}")

    print(f"\nWritten to results/classifier_comparison.json")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
