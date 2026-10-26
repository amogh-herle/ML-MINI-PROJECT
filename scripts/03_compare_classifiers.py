"""
Phase 3: classifier comparison.

Beyond the paper's own Logistic Regression, compare SVM, Random Forest, and
Naive Bayes on the SAME feature vectors (LDA topic vectors at num_topics=100,
Doc2Vec embeddings at vector_size=50 -- the Phase 2 tuned-winning settings),
for both label tasks (issue_area 15-label, issue fine-grained). This isolates
the effect of classifier choice from the effect of feature extraction, which
Phase 1/2 already tuned.

Usage:
    python scripts/01_replicate.py           # must run first (downloads/caches corpus)
    python scripts/02_tune_features.py       # must run first (confirms best feature settings)
    python scripts/03_compare_classifiers.py
"""
import json
import sys
import time

sys.path.insert(0, "src")
from data.pipeline import load_split_tokenize, evaluate_multiclass
from models.lda_lr import LDAFeaturizer
from models.doc2vec_lr import Doc2VecFeaturizer
from models.classifiers import build_svm, build_random_forest, build_naive_bayes, train_predict

SEED = 42
LDA_NUM_TOPICS = 100      # Phase 2 winner
DOC2VEC_VECTOR_SIZE = 50  # Phase 2 winner


def main():
    t0 = time.time()
    print("Loading + splitting corpus (same seed as run_all.py)...")
    data = load_split_tokenize()

    print(f"Building LDA features (num_topics={LDA_NUM_TOPICS}, tuned winner)...")
    lda_feat = LDAFeaturizer(num_topics=LDA_NUM_TOPICS, passes=5, seed=SEED)
    lda_X = (lda_feat.fit_transform(data["train_tok"]), lda_feat.transform(data["dev_tok"]), lda_feat.transform(data["test_tok"]))

    print(f"Building Doc2Vec features (vector_size={DOC2VEC_VECTOR_SIZE}, tuned winner)...")
    d2v_feat = Doc2VecFeaturizer(vector_size=DOC2VEC_VECTOR_SIZE, window=8, epochs=20, seed=SEED)
    d2v_X = (d2v_feat.fit_transform(data["train_tok"]), d2v_feat.transform(data["dev_tok"]), d2v_feat.transform(data["test_tok"]))

    feature_sets = {"lda": lda_X, "doc2vec": d2v_X}
    classifiers = {
        "svm": lambda: build_svm(C=1.0, kernel="rbf"),
        "random_forest": lambda: build_random_forest(n_estimators=200, seed=SEED),
        "naive_bayes": lambda: build_naive_bayes(),
    }

    results = {}
    for feat_name, (Xtr, Xdev, Xtest) in feature_sets.items():
        for task_name in ("issue_area", "issue"):
            ytr, ydev, ytest = data[task_name]
            for clf_name, clf_builder in classifiers.items():
                key = f"{clf_name}_{feat_name}_{task_name}"
                print(f"Training {key}...")
                dev_pred, test_pred = train_predict(clf_builder(), Xtr, ytr, Xdev, Xtest)
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
