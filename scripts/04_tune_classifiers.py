"""
Phase 4: deep-tune the single best model, not every combo.

Phase 3 found SVM + Doc2Vec + issue_area is the overall winner (68.8% test
accuracy). Earlier versions of this phase grid-searched SVM/RandomForest/
NaiveBayes across all 14 feature x task x classifier combinations -- that
full sweep lives in results/classifier_comparison.json (Phase 3) if you
want it; this phase instead goes deep on just the one model that actually
matters for the live demo (predict.py), with a much wider grid than was
affordable across 14 combos.

Selection metric: dev ACCURACY (not macro-F1). An earlier macro-F1-selected
sweep over all 14 combos found that optimizing for macro-F1 can trade away
accuracy (SVM's case: +1.3pp F1 for -5.5pp accuracy) -- for the one model
that's actually deployed in predict.py, we want the headline number to
improve, so we select on it directly here.

Usage:
    python scripts/01_replicate.py       # must run first (downloads/caches corpus)
    python scripts/02_tune_features.py   # must run first (confirms Doc2Vec vector_size=50)
    python scripts/04_tune_classifiers.py
"""
import itertools
import json
import os
import pickle
import sys
import time

sys.path.insert(0, "src")
from data.pipeline import load_split_tokenize, evaluate_multiclass
from models.doc2vec_lr import Doc2VecFeaturizer
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

SEED = 42
VECTOR_SIZE = 50  # Phase 2 winner
CACHE = "results/feature_cache.pkl"

GRID = (
    [{"kernel": "rbf", "C": C, "gamma": g} for C in [0.1, 1, 5, 10, 50, 100] for g in ["scale", "auto", 0.001, 0.01, 0.1]]
    + [{"kernel": "linear", "C": C} for C in [0.01, 0.1, 1, 5, 10, 50]]
    + [{"kernel": "poly", "C": C, "degree": d} for C in [1, 10] for d in [2, 3]]
)


def build_svm(**params):
    return make_pipeline(StandardScaler(), SVC(**params))


def load_doc2vec_features():
    if os.path.exists(CACHE):
        print("Loading cached Doc2Vec features...")
        with open(CACHE, "rb") as f:
            cached = pickle.load(f)
        if cached.get("vector_size") == VECTOR_SIZE:
            return cached["X"], cached["y"]

    print("Loading corpus + training Doc2Vec (vector_size=50)...")
    data = load_split_tokenize()
    d2v = Doc2VecFeaturizer(vector_size=VECTOR_SIZE, window=8, epochs=20, seed=SEED)
    Xtr = d2v.fit_transform(data["train_tok"])
    Xdev, Xtest = d2v.transform(data["dev_tok"]), d2v.transform(data["test_tok"])
    X = (Xtr, Xdev, Xtest)
    y = data["issue_area"]

    os.makedirs("results", exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump({"vector_size": VECTOR_SIZE, "X": X, "y": y}, f)
    return X, y


def main():
    t0 = time.time()
    (Xtr, Xdev, Xtest), (ytr, ydev, ytest) = load_doc2vec_features()

    print(f"Grid-searching SVM ({len(GRID)} configs), selecting on dev accuracy...")
    best_acc, best_params, best_clf = -1, None, None
    for i, params in enumerate(GRID, 1):
        clf = build_svm(**params)
        clf.fit(Xtr, ytr)
        acc = evaluate_multiclass(ydev, clf.predict(Xdev))["accuracy"]
        if acc > best_acc:
            best_acc, best_params, best_clf = acc, params, clf
            print(f"  [{i}/{len(GRID)}] {params} -> dev acc={acc:.4f}  (new best)")

    dev_metrics = evaluate_multiclass(ydev, best_clf.predict(Xdev))
    test_metrics = evaluate_multiclass(ytest, best_clf.predict(Xtest))
    baseline = {"accuracy": 0.6883876357560568, "macro_f1": 0.49927435622074606}  # Phase 3's untuned SVM

    result = {
        "model": "svm_doc2vec_issue_area",
        "best_params": best_params,
        "dev_after": dev_metrics,
        "test_before": baseline,
        "test_after": test_metrics,
    }
    with open("results/tuned_classifiers.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nBest params: {best_params}")
    print(f"Test accuracy: {baseline['accuracy']:.1%} -> {test_metrics['accuracy']:.1%}")
    print(f"Test macro-F1: {baseline['macro_f1']:.3f} -> {test_metrics['macro_f1']:.3f}")
    print(f"\nWritten to results/tuned_classifiers.json")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
