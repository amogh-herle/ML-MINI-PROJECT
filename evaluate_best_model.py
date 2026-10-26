"""
Confusion matrix for the live-demo model (Doc2Vec + SVM, issue_area, the
68.8%-accuracy model highlighted throughout the deck/write-up) on the
held-out test set. Loads the artifacts train_best_model.py saved -- doesn't
retrain anything.

Usage:
    python train_best_model.py      # must run first
    python evaluate_best_model.py
"""
import json
import sys

sys.path.insert(0, "src")
from pipeline import load_split_tokenize
from labels import AREA_NAMES
from gensim.models.doc2vec import Doc2Vec
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

MODEL_DIR = "results/best_model"
NAVY = "#142340"


def main():
    print("Loading saved model...")
    d2v = Doc2Vec.load(f"{MODEL_DIR}/doc2vec.model")
    clf = joblib.load(f"{MODEL_DIR}/svm.joblib")
    meta = joblib.load(f"{MODEL_DIR}/meta.joblib")

    print("Loading test split (same seed as train_best_model.py)...")
    data = load_split_tokenize()
    _, _, ytest = data["issue_area"]

    print("Predicting on test set...")
    Xtest = np.vstack([d2v.infer_vector(t, epochs=meta["epochs"]) for t in data["test_tok"]])
    ypred = clf.predict(Xtest)

    labels = sorted(AREA_NAMES.keys())
    names = [AREA_NAMES[l] for l in labels]
    cm = confusion_matrix(ytest, ypred, labels=labels)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)  # row-normalized: recall per true class

    with open("results/confusion_matrix.json", "w") as f:
        json.dump({"labels": labels, "names": names, "matrix": cm.tolist()}, f, indent=2)
    print("Wrote results/confusion_matrix.json")

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("True category")
    ax.set_title("Confusion Matrix: SVM + Doc2Vec (issue_area, test set)", fontweight="bold", color=NAVY)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j] > 0:
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(im, ax=ax, label="Fraction of true class (row-normalized)")
    plt.tight_layout()
    plt.savefig("results/figures/confusion_matrix.png", dpi=150)
    print("Wrote results/figures/confusion_matrix.png")


if __name__ == "__main__":
    main()
