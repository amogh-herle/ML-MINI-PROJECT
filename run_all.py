"""
Full replication of the reference paper's experiment grid:
"Sentiment Analysis of Chinese Microblog Posts on Attitude Towards Vaccination
Since the Distribution of COVID-19 Vaccines in China" (Bai, He, Yuan — Stanford
CS229, 2021).

Reproduces Table 1 (classifier x tokenization grid: train/dev acc, dev P/R/F1),
Table 2 (test-set performance of the two best dev models), and documents the
Table 3 / Figure 1 trend-analysis limitation (no timestamps in substitute data).

Dataset substitute: the original paper's scraped/hand-labelled Weibo vaccine
dataset was never publicly released, so we use weibo_senti_100k (SophonPlus /
ChineseNlpCorpus) downsampled to the paper's scale (2100/850/850 split). See
README.md for full list of deviations.

Usage:
    python run_all.py
"""
import json
import sys
import time
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, "src")
from preprocessing import preprocess_dataframe, TOKENIZERS
from models import (
    build_naive_bayes, build_logistic_regression, build_svm,
    fit_predict, fit_predict_dense, evaluate,
)
from naive_bayes import NaiveBayesClassifier, InterpolatedNaiveBayes
from bigram_lm import BigramLMClassifier
from embeddings import AveragedWord2VecVectorizer
from lstm_model import train_and_predict as lstm_train_predict

DATA_PATH = "data/weibo_senti_100k.parquet"
RESULTS_DIR = "results"
SEED = 42
N_TRAIN, N_DEV, N_TEST = 2100, 850, 850


def load_split():
    df = pd.read_parquet(DATA_PATH)
    df = preprocess_dataframe(df, text_col="review", label_col="label")
    df = df.sample(n=N_TRAIN + N_DEV + N_TEST, random_state=SEED).reset_index(drop=True)
    train_df, rest = train_test_split(df, train_size=N_TRAIN, random_state=SEED, stratify=df["label"])
    dev_df, test_df = train_test_split(rest, train_size=N_DEV, random_state=SEED, stratify=rest["label"])
    return train_df.reset_index(drop=True), dev_df.reset_index(drop=True), test_df.reset_index(drop=True)


def tok_all(df, name):
    return df["review"].apply(TOKENIZERS[name]).tolist()


def main():
    t0 = time.time()
    print("Loading + splitting data (paper scale: 2100/850/850)...")
    train_df, dev_df, test_df = load_split()
    ytr, ydev, ytest = train_df["label"].tolist(), dev_df["label"].tolist(), test_df["label"].tolist()

    # Pre-tokenize once per scheme (reused across classifiers)
    tokenized = {
        split: {name: tok_all(df, name) for name in ["unigram", "bigram", "trigram", "jieba"]}
        for split, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]
    }

    table1 = {}  # table1[classifier][tokenization] = {train, dev}

    # ---------------- Naive Bayes (Sec 4.1): unigram/bigram/trigram/jieba ----------------
    print("Naive Bayes...")
    table1["naive_bayes"] = {}
    for tok_name in ["unigram", "bigram", "trigram", "jieba"]:
        nb = NaiveBayesClassifier().fit(tokenized["train"][tok_name], ytr)
        train_pred = nb.predict(tokenized["train"][tok_name])
        dev_pred = nb.predict(tokenized["dev"][tok_name])
        table1["naive_bayes"][tok_name] = {
            "train_acc": evaluate(ytr, train_pred)["accuracy"],
            "dev": evaluate(ydev, dev_pred),
        }
    # Linear interpolation of bigram+trigram NB (eq. 3), lambda sweep like the paper
    print("Naive Bayes (bigram+trigram interpolation)...")
    best_lam, best_f1 = None, -1
    for lam in [0.2, 0.5, 0.8]:
        inb = InterpolatedNaiveBayes(lam=lam).fit(tokenized["train"]["bigram"], tokenized["train"]["trigram"], ytr)
        dev_pred = inb.predict(tokenized["dev"]["bigram"], tokenized["dev"]["trigram"])
        f1 = evaluate(ydev, dev_pred)["f1"]
        if f1 > best_f1:
            best_f1, best_lam = f1, lam
    inb = InterpolatedNaiveBayes(lam=best_lam).fit(tokenized["train"]["bigram"], tokenized["train"]["trigram"], ytr)
    train_pred = inb.predict(tokenized["train"]["bigram"], tokenized["train"]["trigram"])
    dev_pred = inb.predict(tokenized["dev"]["bigram"], tokenized["dev"]["trigram"])
    table1["naive_bayes"]["bigram_trigram_interpolated"] = {
        "lambda": best_lam,
        "train_acc": evaluate(ytr, train_pred)["accuracy"],
        "dev": evaluate(ydev, dev_pred),
    }

    # ---------------- Bigram Language Model (Sec 4.4): unigram/jieba only ----------------
    print("Bigram language model...")
    table1["bigram_model"] = {}
    for tok_name in ["unigram", "jieba"]:
        blm = BigramLMClassifier().fit(tokenized["train"][tok_name], ytr)
        train_pred = blm.predict(tokenized["train"][tok_name])
        dev_pred = blm.predict(tokenized["dev"][tok_name])
        table1["bigram_model"][tok_name] = {
            "train_acc": evaluate(ytr, train_pred)["accuracy"],
            "dev": evaluate(ydev, dev_pred),
        }

    # ---------------- SVM (Sec 4.2): unigram/bigram/trigram/jieba/text2vec ----------------
    print("SVM...")
    table1["svm"] = {}
    for tok_name in ["unigram", "bigram", "trigram", "jieba"]:
        vec, clf = build_svm()
        train_pred, dev_pred, _ = fit_predict(vec, clf, tokenized["train"][tok_name], ytr, tokenized["dev"][tok_name])
        table1["svm"][tok_name] = {
            "train_acc": evaluate(ytr, train_pred)["accuracy"],
            "dev": evaluate(ydev, dev_pred),
        }
    print("SVM (text2vec substitute: averaged word2vec)...")
    w2v = AveragedWord2VecVectorizer(vector_size=100)
    Xtr_vec = w2v.fit_transform(tokenized["train"]["jieba"])
    Xdev_vec = w2v.transform(tokenized["dev"]["jieba"])
    Xtest_vec = w2v.transform(tokenized["test"]["jieba"])
    _, svm_clf = build_svm()
    train_pred, dev_pred, test_pred_svm_t2v = fit_predict_dense(svm_clf, Xtr_vec, ytr, Xdev_vec, Xtest_vec)
    table1["svm"]["text2vec"] = {
        "train_acc": evaluate(ytr, train_pred)["accuracy"],
        "dev": evaluate(ydev, dev_pred),
    }

    # ---------------- Logistic Regression (Sec 4.3): unigram/bigram/trigram/jieba/text2vec ----------------
    print("Logistic Regression (per-tokenization C search on dev, as in paper)...")
    table1["logistic_regression"] = {}
    for tok_name in ["unigram", "bigram", "trigram", "jieba"]:
        best_C, best_f1, best_pack = None, -1, None
        for C in [0.1, 0.5, 1.0, 2.0, 5.0]:
            vec, clf = build_logistic_regression(C=C)
            train_pred, dev_pred, _ = fit_predict(vec, clf, tokenized["train"][tok_name], ytr, tokenized["dev"][tok_name])
            f1 = evaluate(ydev, dev_pred)["f1"]
            if f1 > best_f1:
                best_f1, best_C = f1, C
                best_pack = (train_pred, dev_pred)
        table1["logistic_regression"][tok_name] = {
            "best_C": best_C,
            "train_acc": evaluate(ytr, best_pack[0])["accuracy"],
            "dev": evaluate(ydev, best_pack[1]),
        }
    print("Logistic Regression (text2vec substitute)...")
    best_C, best_f1, best_pack = None, -1, None
    for C in [0.1, 0.5, 1.0, 2.0, 5.0]:
        clf = build_logistic_regression(C=C)[1]
        train_pred, dev_pred, test_pred = fit_predict_dense(clf, Xtr_vec, ytr, Xdev_vec, Xtest_vec)
        f1 = evaluate(ydev, dev_pred)["f1"]
        if f1 > best_f1:
            best_f1, best_C = f1, C
            best_pack = (train_pred, dev_pred, test_pred)
    table1["logistic_regression"]["text2vec"] = {
        "best_C": best_C,
        "train_acc": evaluate(ytr, best_pack[0])["accuracy"],
        "dev": evaluate(ydev, best_pack[1]),
    }
    test_pred_lr_t2v = best_pack[2]

    # ---------------- LSTM (Sec 4.5): unigram/bigram/trigram/jieba ----------------
    print("LSTM (this takes the longest)...")
    table1["lstm"] = {}
    lstm_test_preds = {}
    for tok_name in ["unigram", "bigram", "trigram", "jieba"]:
        train_pred, dev_pred, test_pred = lstm_train_predict(
            tokenized["train"][tok_name], ytr, tokenized["dev"][tok_name], ydev,
            test_tokens=tokenized["test"][tok_name], epochs=8,
        )
        table1["lstm"][tok_name] = {
            "train_acc": evaluate(ytr, train_pred)["accuracy"],
            "dev": evaluate(ydev, dev_pred),
        }
        lstm_test_preds[tok_name] = test_pred

    with open(f"{RESULTS_DIR}/table1.json", "w") as f:
        json.dump(table1, f, indent=2, ensure_ascii=False)
    print(f"Table 1 written to {RESULTS_DIR}/table1.json")

    # ---------------- Table 2: pick best 2 models by dev F1, evaluate on test ----------------
    print("Selecting best 2 models by dev F1 for Table 2...")
    flat = []
    for clf_name, toks in table1.items():
        for tok_name, metrics in toks.items():
            flat.append((clf_name, tok_name, metrics["dev"]["f1"]))
    flat.sort(key=lambda x: -x[2])
    top2 = flat[:2]
    print("Top 2 by dev F1:", top2)

    table2 = {}
    for clf_name, tok_name, _ in top2:
        key = f"{clf_name}_{tok_name}"
        if clf_name == "svm" and tok_name == "text2vec":
            test_pred = test_pred_svm_t2v
        elif clf_name == "logistic_regression" and tok_name == "text2vec":
            test_pred = test_pred_lr_t2v
        elif clf_name == "lstm":
            test_pred = lstm_test_preds[tok_name]
        elif clf_name == "naive_bayes":
            nb = NaiveBayesClassifier().fit(tokenized["train"][tok_name], ytr)
            test_pred = nb.predict(tokenized["test"][tok_name])
        elif clf_name == "bigram_model":
            blm = BigramLMClassifier().fit(tokenized["train"][tok_name], ytr)
            test_pred = blm.predict(tokenized["test"][tok_name])
        elif clf_name == "svm":
            vec, clf = build_svm()
            _, _, test_pred = fit_predict(vec, clf, tokenized["train"][tok_name], ytr, tokenized["dev"][tok_name], tokenized["test"][tok_name])
        elif clf_name == "logistic_regression":
            best_C = table1["logistic_regression"][tok_name]["best_C"]
            vec, clf = build_logistic_regression(C=best_C)
            _, _, test_pred = fit_predict(vec, clf, tokenized["train"][tok_name], ytr, tokenized["dev"][tok_name], tokenized["test"][tok_name])
        else:
            test_pred = None
        table2[key] = evaluate(ytest, test_pred) if test_pred is not None else None

    with open(f"{RESULTS_DIR}/table2.json", "w") as f:
        json.dump(table2, f, indent=2, ensure_ascii=False)
    print(json.dumps(table2, indent=2, ensure_ascii=False))
    print(f"Table 2 written to {RESULTS_DIR}/table2.json")

    print("\nNOTE (Table 3 / trend analysis): the substitute dataset (weibo_senti_100k) "
          "has no post timestamps, so the paper's month-by-month positive-sentiment trend "
          "(Table 3, Figure 1) cannot be replicated. Documented as a limitation in README.md.")

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
