"""
Classify ONE new piece of legal text into its issue_area category, using the
saved Doc2Vec + SVM model (see train_best_model.py). This is the live-demo
script: paste in a sentence or two from a court opinion and see the
predicted category, with confidence scores for the top 3.

Usage:
    python train_best_model.py          # once, trains + saves the model (a few minutes)
    python predict.py "your legal text here"
    python predict.py --file path/to/opinion.txt
    python predict.py                    # no argument: drops into an interactive prompt
"""
import os
import sys

sys.path.insert(0, "src")
from preprocessing import simple_tokenize
from labels import area_name
from gensim.models.doc2vec import Doc2Vec
import joblib

MODEL_DIR = "results/best_model"


def load_model():
    if not os.path.exists(f"{MODEL_DIR}/svm.joblib"):
        print("No trained model found. Run `python train_best_model.py` first (takes a few minutes).")
        sys.exit(1)
    d2v = Doc2Vec.load(f"{MODEL_DIR}/doc2vec.model")
    clf = joblib.load(f"{MODEL_DIR}/svm.joblib")
    meta = joblib.load(f"{MODEL_DIR}/meta.joblib")
    return d2v, clf, meta


def classify(text, d2v, clf, meta, top_k=3):
    tokens = simple_tokenize(text)
    if not tokens:
        print("Warning: no recognizable words after cleaning (stopwords/punctuation removed) — "
              "prediction will be unreliable.")
    vec = d2v.infer_vector(tokens, epochs=meta["epochs"]).reshape(1, -1)
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
    d2v, clf, meta = load_model()

    if len(sys.argv) > 1 and sys.argv[1] == "--file":
        with open(sys.argv[2]) as f:
            # strip '#'-prefixed comment lines (e.g. extract_samples.py's ground-truth
            # header) so they can't leak into the prediction
            text = "\n".join(line for line in f if not line.lstrip().startswith("#"))
        print_result(text, classify(text, d2v, clf, meta))
    elif len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        print_result(text, classify(text, d2v, clf, meta))
    else:
        print("Paste or type legal text, then press Enter (Ctrl+C to quit):\n")
        while True:
            try:
                text = input("> ")
            except (KeyboardInterrupt, EOFError):
                print()
                break
            if text.strip():
                print_result(text, classify(text, d2v, clf, meta))


if __name__ == "__main__":
    main()
