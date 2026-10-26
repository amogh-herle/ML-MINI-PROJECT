"""
Train and save the best single model for classifying one new document —
Doc2Vec (vector_size=50, Phase 2's winning setting) + SVM (RBF, C=1.0,
Phase 3's best accuracy: 68.8% test accuracy on issue_area). Saved once here
so predict.py can load it instantly instead of retraining Doc2Vec (a few
minutes) on every single prediction.

Usage:
    python train_best_model.py
    python predict.py "some legal text..."
"""
import sys
import time

sys.path.insert(0, "src")
from pipeline import load_split_tokenize, evaluate_multiclass
from doc2vec_lr import Doc2VecFeaturizer
from classifiers import build_svm

SEED = 42
VECTOR_SIZE = 50
MODEL_DIR = "results/best_model"


def main():
    t0 = time.time()
    print("Loading + splitting corpus...")
    data = load_split_tokenize()
    ytr, ydev, ytest = data["issue_area"]

    print(f"Training Doc2Vec (vector_size={VECTOR_SIZE})...")
    d2v = Doc2VecFeaturizer(vector_size=VECTOR_SIZE, window=8, epochs=20, seed=SEED)
    Xtr = d2v.fit_transform(data["train_tok"])
    Xdev, Xtest = d2v.transform(data["dev_tok"]), d2v.transform(data["test_tok"])

    print("Training SVM (RBF, C=1.0, probability=True for confidence scores)...")
    clf = build_svm(C=1.0, kernel="rbf", probability=True)
    clf.fit(Xtr, ytr)

    dev_metrics = evaluate_multiclass(ydev, clf.predict(Xdev))
    test_metrics = evaluate_multiclass(ytest, clf.predict(Xtest))
    print(f"Dev accuracy:  {dev_metrics['accuracy']:.1%}")
    print(f"Test accuracy: {test_metrics['accuracy']:.1%}  (matches Phase 3's reported 68.8%)")

    import os
    import joblib
    os.makedirs(MODEL_DIR, exist_ok=True)
    d2v.model.save(f"{MODEL_DIR}/doc2vec.model")
    joblib.dump(clf, f"{MODEL_DIR}/svm.joblib")
    joblib.dump({"vector_size": VECTOR_SIZE, "epochs": d2v.epochs}, f"{MODEL_DIR}/meta.joblib")
    print(f"\nSaved to {MODEL_DIR}/ — run `python predict.py \"your text\"` to classify a new document.")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
