"""
Phase 5: pretrained Sentence-BERT embeddings vs. Doc2Vec.

No training step for the featurizer -- SBERT is pretrained on ~1B+ sentence
pairs before it ever sees a single court opinion, unlike Doc2Vec (Phases
1-4), which learns from scratch on our ~5,600 training documents alone.
Same comparison protocol as the earlier phases: issue_area + issue, Logistic
Regression and SVM (the paper's classifier and Phase 3's winner) on top.

Usage:
    python scripts/05_sbert.py
"""
import json
import sys
import time

sys.path.insert(0, "src")
from data.pipeline import load_split_tokenize, evaluate_multiclass
from models.sbert_lr import SBERTFeaturizer, train_sbert_lr
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def main():
    t0 = time.time()
    print("Loading + splitting corpus...", flush=True)
    data = load_split_tokenize()

    print("Loading pretrained SBERT (all-MiniLM-L6-v2) and encoding documents "
          "(no training -- just inference)...", flush=True)
    sbert = SBERTFeaturizer()
    Xtr = sbert.fit_transform(data["train_text"])
    Xdev = sbert.transform(data["dev_text"])
    Xtest = sbert.transform(data["test_text"])

    results = {}
    for task in ("issue_area", "issue"):
        ytr, ydev, ytest = data[task]

        print(f"SBERT + Logistic Regression: {task}...", flush=True)
        clf = train_sbert_lr(Xtr, ytr, C=1.0)
        results[f"sbert_lr_{task}"] = {
            "dev": evaluate_multiclass(ydev, clf.predict(Xdev)),
            "test": evaluate_multiclass(ytest, clf.predict(Xtest)),
        }

        print(f"SBERT + SVM: {task}...", flush=True)
        svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0))
        svm.fit(Xtr, ytr)
        results[f"sbert_svm_{task}"] = {
            "dev": evaluate_multiclass(ydev, svm.predict(Xdev)),
            "test": evaluate_multiclass(ytest, svm.predict(Xtest)),
        }

    with open("results/sbert_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n--- Summary (test accuracy) ---", flush=True)
    for key, r in sorted(results.items(), key=lambda kv: -kv[1]["test"]["accuracy"]):
        print(f"  {key:25s} acc={r['test']['accuracy']:.4f}  macro_f1={r['test']['macro_f1']:.4f}", flush=True)
    print(f"\nWritten to results/sbert_results.json", flush=True)
    print(f"Total runtime: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
