"""
Phase 4 (light): quick hyperparameter tuning for SVM, Random Forest, Naive Bayes.

Same protocol as tune_all.py (select on DEV macro-F1, test predicted once),
but with small grids, cached features and progress output.

Usage:
    python tune_classifiers_light.py
"""
import itertools
import json
import os
import pickle
import sys
import time

sys.path.insert(0, "src")
from preprocessing import load_corpus, split_records, simple_tokenize
from lda_lr import LDAFeaturizer
from doc2vec_lr import Doc2VecFeaturizer
from run_all import filter_rare_classes, evaluate_multiclass

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

SEED = 42
LDA_NUM_TOPICS = 100
DOC2VEC_VECTOR_SIZE = 50
CACHE = "results/feature_cache.pkl"


def expand(grid):
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*grid.values())]


def make_svm(kernel, C, scale=True):
    svc = SVC(kernel=kernel, C=C, gamma="scale") if kernel == "rbf" else SVC(kernel=kernel, C=C)
    return make_pipeline(StandardScaler(), svc) if scale else svc


def make_rf(n_estimators, max_depth, min_samples_leaf):
    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, max_features="sqrt",
        n_jobs=-1, random_state=SEED,
    )


def make_gnb(var_smoothing):
    return GaussianNB(var_smoothing=var_smoothing)


def make_mnb(alpha):
    return MultinomialNB(alpha=alpha)


# ---------------- LIGHT grids ----------------
SVM_GRID = expand({"kernel": ["rbf"], "C": [0.1, 1, 10]}) + expand({"kernel": ["linear"], "C": [0.1, 1]})
RF_GRID = expand({"n_estimators": [100, 200], "max_depth": [None, 20], "min_samples_leaf": [1, 3]})
GNB_GRID = expand({"var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-1]})
MNB_GRID = expand({"alpha": [0.01, 0.1, 1.0, 5.0]})  # LDA only (non-negative)


def build_features():
    if os.path.exists(CACHE):
        print("Loading cached features...")
        with open(CACHE, "rb") as f:
            return pickle.load(f)

    print("Loading + splitting corpus...")
    records = filter_rare_classes(load_corpus())
    train, dev, test = split_records(records)

    train_tok = [simple_tokenize(r["text"]) for r in train]
    dev_tok = [simple_tokenize(r["text"]) for r in dev]
    test_tok = [simple_tokenize(r["text"]) for r in test]

    print(f"Building LDA features (num_topics={LDA_NUM_TOPICS}) - this takes a few minutes, only once...")
    lda = LDAFeaturizer(num_topics=LDA_NUM_TOPICS, passes=5, seed=SEED)
    lda_X = (lda.fit_transform(train_tok), lda.transform(dev_tok), lda.transform(test_tok))

    print(f"Building Doc2Vec features (vector_size={DOC2VEC_VECTOR_SIZE}) - also once...")
    d2v = Doc2VecFeaturizer(vector_size=DOC2VEC_VECTOR_SIZE, window=8, epochs=20, seed=SEED)
    d2v_X = (d2v.fit_transform(train_tok), d2v.transform(dev_tok), d2v.transform(test_tok))

    data = {
        "features": {"lda": lda_X, "doc2vec": d2v_X},
        "labels": {
            "issue_area": ([r["issue_area"] for r in train], [r["issue_area"] for r in dev], [r["issue_area"] for r in test]),
            "issue": ([r["issue"] for r in train], [r["issue"] for r in dev], [r["issue"] for r in test]),
        },
    }
    os.makedirs("results", exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(data, f)
    print("Features cached.")
    return data


def tune(name, builder, grid, Xtr, ytr, Xdev, ydev, Xtest, ytest):
    best_f1, best_params, best_clf = -1, None, None
    for i, params in enumerate(grid, 1):
        t = time.time()
        try:
            clf = builder(**params)
            clf.fit(Xtr, ytr)
            f1 = evaluate_multiclass(ydev, clf.predict(Xdev))["macro_f1"]
        except Exception as e:
            print(f"    [{i}/{len(grid)}] skip {params}: {e}", flush=True)
            continue
        print(f"    [{i}/{len(grid)}] {params} -> dev f1={f1:.4f} ({time.time() - t:.1f}s)", flush=True)
        if f1 > best_f1:
            best_f1, best_params, best_clf = f1, params, clf
    if best_clf is None:
        print(f"    WARNING: every config in {name}'s grid raised an exception -- skipping", flush=True)
        return {"best_params": None, "error": "all configs failed"}
    return {
        "best_params": best_params,
        "dev_after": evaluate_multiclass(ydev, best_clf.predict(Xdev)),
        "test_after": evaluate_multiclass(ytest, best_clf.predict(Xtest)),  # test touched once
    }


def main():
    t0 = time.time()
    data = build_features()

    baseline = {}
    if os.path.exists("results/classifier_comparison.json"):
        with open("results/classifier_comparison.json") as f:
            baseline = json.load(f)

    results = {}
    for feat_name, (Xtr, Xdev, Xtest) in data["features"].items():
        for task_name, (ytr, ydev, ytest) in data["labels"].items():
            print(f"\n=== {feat_name} / {task_name} ===", flush=True)
            jobs = [
                ("svm", make_svm, SVM_GRID),
                ("random_forest", make_rf, RF_GRID),
                ("naive_bayes_gaussian", make_gnb, GNB_GRID),
            ]
            if feat_name == "lda":
                jobs.append(("naive_bayes_multinomial", make_mnb, MNB_GRID))

            for clf_name, builder, grid in jobs:
                print(f"  tuning {clf_name} ({len(grid)} configs)...", flush=True)
                r = tune(clf_name, builder, grid, Xtr, ytr, Xdev, ydev, Xtest, ytest)
                # classifier_comparison.json's baseline only has a Gaussian NB entry
                # (key "naive_bayes_*"); there is no Multinomial NB baseline to
                # compare against, so don't attach one -- that would silently
                # compare two different algorithms as if they were "before/after".
                if clf_name != "naive_bayes_multinomial":
                    base_name = clf_name.replace("_gaussian", "")
                    base_key = f"{base_name}_{feat_name}_{task_name}"
                    if base_key in baseline:
                        r["test_before"] = baseline[base_key]["test"]
                results[f"{clf_name}_{feat_name}_{task_name}"] = r

    with open("results/tuned_classifiers.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print("\n--- Summary (test macro_f1, before -> after) ---")
    failed = {k: r for k, r in results.items() if "test_after" not in r}
    ok = {k: r for k, r in results.items() if "test_after" in r}
    for key, r in sorted(ok.items(), key=lambda kv: -kv[1]["test_after"]["macro_f1"]):
        after = r["test_after"]
        before = r.get("test_before")
        b = f"{before['macro_f1']:.4f}" if before else "  n/a "
        print(f"  {key:50s} {b} -> {after['macro_f1']:.4f}  (acc={after['accuracy']:.4f})")
    for key, r in failed.items():
        print(f"  {key:50s} FAILED: {r.get('error', 'unknown error')}")

    print("\nWritten to results/tuned_classifiers.json")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()