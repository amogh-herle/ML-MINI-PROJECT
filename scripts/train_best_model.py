"""
Train and save the best single model for classifying one new document --
SBERT (all-MiniLM-L6-v2, pretrained, Phase 5's winner: 70.1% test accuracy,
beating every Doc2Vec variant including Phase 4's deep-tuned SVM) + SVM.
No training for the embedding step -- SBERT is pretrained, we only fit the
SVM on top. Saved once here so predict.py doesn't re-encode the training set
on every single prediction.

Usage:
    python scripts/05_sbert.py          # optional but recommended: confirms SBERT is the winner
    python scripts/train_best_model.py
    python scripts/predict.py "some legal text..."
"""
import os
import sys
import time

sys.path.insert(0, "src")
from data.pipeline import load_split_tokenize, evaluate_multiclass
from models.sbert_lr import SBERTFeaturizer, MODEL_NAME
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

MODEL_DIR = "results/best_model"


def main():
    t0 = time.time()
    print("Loading + splitting corpus...")
    data = load_split_tokenize()
    ytr, ydev, ytest = data["issue_area"]

    print(f"Loading pretrained SBERT ({MODEL_NAME}) and encoding documents...")
    sbert = SBERTFeaturizer()
    Xtr = sbert.fit_transform(data["train_text"])
    Xdev = sbert.transform(data["dev_text"])
    Xtest = sbert.transform(data["test_text"])

    print("Training SVM (RBF, C=1.0, probability=True for confidence scores)...")
    clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, probability=True))
    clf.fit(Xtr, ytr)

    dev_metrics = evaluate_multiclass(ydev, clf.predict(Xdev))
    test_metrics = evaluate_multiclass(ytest, clf.predict(Xtest))
    print(f"Dev accuracy:  {dev_metrics['accuracy']:.1%}")
    print(f"Test accuracy: {test_metrics['accuracy']:.1%}")

    import joblib
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, f"{MODEL_DIR}/svm.joblib")
    joblib.dump({"model_name": MODEL_NAME}, f"{MODEL_DIR}/meta.joblib")
    print(f"\nSaved to {MODEL_DIR}/ -- run `python scripts/predict.py \"your text\"` to classify a new document.")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
