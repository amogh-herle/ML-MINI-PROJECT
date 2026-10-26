"""
Pull a handful of real court opinions from the held-out TEST split — one per
a spread of issue_area categories — and save them as .txt files you can feed
straight into predict.py to try the model on real, unseen text.

Each file is named <code>_<category>.txt and starts with a comment line
giving the ground-truth label, so you can check predict.py's guess against
the real answer.

Usage:
    python extract_samples.py            # writes samples/ (5 by default)
    python extract_samples.py --n 8      # more samples
    python predict.py --file samples/01_criminal_procedure.txt
"""
import argparse
import os
import re
import sys

sys.path.insert(0, "src")
from pipeline import filter_rare_classes
from preprocessing import load_corpus, split_records
from labels import area_name

OUT_DIR = "samples"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="number of sample opinions to extract")
    args = ap.parse_args()

    print("Loading corpus and reproducing the test split...")
    records = filter_rare_classes(load_corpus())
    _, _, test = split_records(records)

    # one per distinct issue_area, spread across categories, in a stable order
    by_area = {}
    for r in test:
        by_area.setdefault(r["issue_area"], r)
    picked = list(by_area.values())[: args.n]

    os.makedirs(OUT_DIR, exist_ok=True)
    for i, r in enumerate(picked, 1):
        name = area_name(r["issue_area"])
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        path = f"{OUT_DIR}/{i:02d}_{slug}.txt"
        with open(path, "w") as f:
            f.write(f"# Ground truth: {name} (issue_area code {r['issue_area']})\n")
            f.write(r["text"])
        print(f"  wrote {path}  (true label: {name})")

    print(f"\n{len(picked)} samples in {OUT_DIR}/. Try:")
    for i in range(1, len(picked) + 1):
        import glob
        f = glob.glob(f"{OUT_DIR}/{i:02d}_*.txt")[0]
        print(f"  python predict.py --file {f}")


if __name__ == "__main__":
    main()
