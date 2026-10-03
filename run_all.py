"""
Mini-project pipeline: replicate then tune, following the reference paper
"Sentiment Analysis of Chinese Microblog Posts on Attitude Towards Vaccination
Since the Distribution of COVID-19 Vaccines in China" (Bai, He, Yuan — Stanford
CS229, 2021).

Dataset substitute: the original paper's scraped/hand-labelled Weibo vaccine
dataset was never publicly released, so we use weibo_senti_100k (SophonPlus /
ChineseNlpCorpus) as a stand-in, downsampled to the paper's scale
(~2100 train / 850 dev / 850 test) to stay in the same small-data regime.
See README.md for details and deviations.

Usage:
    python run_all.py
"""
import json
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, "src")
from preprocessing import preprocess_dataframe, TOKENIZERS
from models import build_naive_bayes, build_logistic_regression, fit_predict, evaluate

DATA_PATH = "data/weibo_senti_100k.parquet"
RESULTS_PATH = "results/results.json"
SEED = 42

# Paper's split sizes
N_TRAIN, N_DEV, N_TEST = 2100, 850, 850


def load_split():
    df = pd.read_parquet(DATA_PATH)
    df = preprocess_dataframe(df, text_col="review", label_col="label")
    df = df.sample(n=N_TRAIN + N_DEV + N_TEST, random_state=SEED).reset_index(drop=True)

    train_df, rest = train_test_split(
        df, train_size=N_TRAIN, random_state=SEED, stratify=df["label"]
    )
    dev_df, test_df = train_test_split(
        rest, train_size=N_DEV, random_state=SEED, stratify=rest["label"]
    )
    return train_df, dev_df, test_df


def tokenize_all(df, tokenizer_fn):
    return df["review"].apply(tokenizer_fn).tolist()


def run_phase1(train_df, dev_df, test_df):
    """Phase 1: replicate paper's NB + LR baselines with its default settings,
    no hyperparameter search, bigram tokenization (paper's best NB setting)."""
    results = {}
    tok_fn = TOKENIZERS["bigram"]

    Xtr = tokenize_all(train_df, tok_fn)
    Xdev = tokenize_all(dev_df, tok_fn)
    Xtest = tokenize_all(test_df, tok_fn)
    ytr, ydev, ytest = train_df["label"].tolist(), dev_df["label"].tolist(), test_df["label"].tolist()

    # Naive Bayes (paper default: Laplace smoothing, alpha=1.0)
    vec, clf = build_naive_bayes(alpha=1.0)
    train_pred, dev_pred, test_pred = fit_predict(vec, clf, Xtr, ytr, Xdev, Xtest)
    results["naive_bayes_untuned"] = {
        "train": evaluate(ytr, train_pred),
        "dev": evaluate(ydev, dev_pred),
        "test": evaluate(ytest, test_pred),
    }

    # Logistic Regression (paper default: C=2 for jieba tokenization; we use bigram here, C=2)
    vec, clf = build_logistic_regression(C=2.0)
    train_pred, dev_pred, test_pred = fit_predict(vec, clf, Xtr, ytr, Xdev, Xtest)
    results["logistic_regression_untuned"] = {
        "train": evaluate(ytr, train_pred),
        "dev": evaluate(ydev, dev_pred),
        "test": evaluate(ytest, test_pred),
    }
    return results, (Xtr, Xdev, Xtest, ytr, ydev, ytest)


def run_phase2(cached, test_df):
    """Phase 2: tune hyperparameters on dev set only (Naive Bayes alpha,
    Logistic Regression C), touch test set once with the best config."""
    Xtr, Xdev, Xtest, ytr, ydev, ytest = cached
    results = {}

    best_alpha, best_f1 = 1.0, -1
    for alpha in [0.1, 0.5, 1.0, 2.0]:
        vec, clf = build_naive_bayes(alpha=alpha)
        _, dev_pred, _ = fit_predict(vec, clf, Xtr, ytr, Xdev)
        f1 = evaluate(ydev, dev_pred)["f1"]
        if f1 > best_f1:
            best_f1, best_alpha = f1, alpha
    vec, clf = build_naive_bayes(alpha=best_alpha)
    train_pred, dev_pred, test_pred = fit_predict(vec, clf, Xtr, ytr, Xdev, Xtest)
    results["naive_bayes_tuned"] = {
        "best_alpha": best_alpha,
        "train": evaluate(ytr, train_pred),
        "dev": evaluate(ydev, dev_pred),
        "test": evaluate(ytest, test_pred),
    }

    best_C, best_f1 = 2.0, -1
    for C in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        vec, clf = build_logistic_regression(C=C)
        _, dev_pred, _ = fit_predict(vec, clf, Xtr, ytr, Xdev)
        f1 = evaluate(ydev, dev_pred)["f1"]
        if f1 > best_f1:
            best_f1, best_C = f1, C
    vec, clf = build_logistic_regression(C=best_C)
    train_pred, dev_pred, test_pred = fit_predict(vec, clf, Xtr, ytr, Xdev, Xtest)
    results["logistic_regression_tuned"] = {
        "best_C": best_C,
        "train": evaluate(ytr, train_pred),
        "dev": evaluate(ydev, dev_pred),
        "test": evaluate(ytest, test_pred),
    }
    return results


def main():
    print("Loading and splitting data (paper-scale: 2100/850/850)...")
    train_df, dev_df, test_df = load_split()

    print("Phase 1: replicating paper's baseline models (bigram tokenization)...")
    phase1_results, cached = run_phase1(train_df, dev_df, test_df)

    print("Phase 2: tuning hyperparameters on dev set...")
    phase2_results = run_phase2(cached, test_df)

    all_results = {**phase1_results, **phase2_results}
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\nResults written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
