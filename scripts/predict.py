"""
Classify ONE new piece of legal text into its issue_area category, using the
saved SBERT + SVM model (see train_best_model.py) -- our best model overall
(70.1% test accuracy, Phase 5). This is the live-demo script: paste in a
sentence or two from a court opinion and see the predicted category, with
confidence scores for the top 3.

Usage:
    python scripts/train_best_model.py          # once, trains + saves the model (a few minutes)
    python scripts/predict.py "your legal text here"
    python scripts/predict.py --file path/to/opinion.txt
    python scripts/predict.py                    # no argument: drops into an interactive prompt
"""
import os
import sys

sys.path.insert(0, "src")
from data.labels import area_name
from sentence_transformers import SentenceTransformer
import joblib

MODEL_DIR = "results/best_model"


def load_model():
    if not os.path.exists(f"{MODEL_DIR}/svm.joblib"):
        print("No trained model found. Run `python scripts/train_best_model.py` first (takes a few minutes).")
        sys.exit(1)
    meta = joblib.load(f"{MODEL_DIR}/meta.joblib")
    sbert = SentenceTransformer(meta["model_name"])
    clf = joblib.load(f"{MODEL_DIR}/svm.joblib")
    return sbert, clf


def classify(text, sbert, clf, top_k=3):
    vec = sbert.encode([text], show_progress_bar=False)
    proba = clf.predict_proba(vec)[0]
    ranked = sorted(zip(clf.classes_, proba), key=lambda kv: -kv[1])[:top_k]
    return ranked


def print_result(text, ranked):
    print(f"\nInput: {text[:200]}{'...' if len(text) > 200 else ''}\n")
    print("Predicted category:")
    for rank, (code, p) in enumerate(ranked, 1):
        marker = "→" if rank == 1 else " "
        print(f"  {marker} {area_name(code):<22s} {p:6.1%}  (code {code})")
    print()


def main():
    sbert, clf = load_model()

    if len(sys.argv) > 1 and sys.argv[1] == "--file":
        with open(sys.argv[2]) as f:
            # strip '#'-prefixed comment lines (e.g. extract_samples.py's ground-truth
            # header) so they can't leak into the prediction
            text = "\n".join(line for line in f if not line.lstrip().startswith("#"))
        print_result(text, classify(text, sbert, clf))
    elif len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        print_result(text, classify(text, sbert, clf))
    else:
        print("Paste or type legal text, then press Enter (Ctrl+C to quit):\n")
        while True:
            try:
                text = input("> ")
            except (KeyboardInterrupt, EOFError):
                print()
                break
            if text.strip():
                print_result(text, classify(text, sbert, clf))


if __name__ == "__main__":
    main()
