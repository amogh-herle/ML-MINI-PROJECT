"""
Phase 2: hyperparameter tuning.

Starting from Phase 1's replication (results/table1.json — the paper's own
settings, no tuning), we now tune each model family's hyperparameters using
the DEV set only, picking the best config, then touch the TEST set exactly
once with that final config. The test set is never used for model/hyperparam
selection.

For each classifier, the tokenization is fixed to whichever one scored
highest dev F1 in Phase 1 (results/table1.json) — hyperparameters are then
tuned around that fixed tokenization choice.

Usage:
    python run_all.py      # must run first, produces results/table1.json
    python tune_all.py
"""
import json
import sys
import time
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, "src")
from preprocessing import preprocess_dataframe, TOKENIZERS
from models import build_logistic_regression, build_svm, fit_predict, fit_predict_dense, evaluate
from naive_bayes import NaiveBayesClassifier
from bigram_lm import BigramLMClassifier
from embeddings import AveragedWord2VecVectorizer
from lstm_model import train_and_predict as lstm_train_predict

DATA_PATH = "data/weibo_senti_100k.parquet"
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


def best_tokenization(table1, clf_name, exclude=()):
    toks = {k: v for k, v in table1[clf_name].items() if k not in exclude}
    return max(toks.items(), key=lambda kv: kv[1]["dev"]["f1"])[0]


def main():
    t0 = time.time()
    with open("results/table1.json") as f:
        table1 = json.load(f)

    print("Loading + splitting data (same seed as run_all.py)...")
    train_df, dev_df, test_df = load_split()
    ytr, ydev, ytest = train_df["label"].tolist(), dev_df["label"].tolist(), test_df["label"].tolist()

    tuned = {}

    # ---------------- Naive Bayes: tune alpha ----------------
    # exclude the interpolated pseudo-tokenization: alpha tuning below applies
    # to a single plain NaiveBayesClassifier on one real tokenization scheme
    tok_name = best_tokenization(table1, "naive_bayes", exclude=("bigram_trigram_interpolated",))
    print(f"Naive Bayes: best tokenization from Phase 1 = {tok_name}; tuning alpha...")
    Xtr = tok_all(train_df, tok_name)
    Xdev = tok_all(dev_df, tok_name)
    Xtest = tok_all(test_df, tok_name)
    best_alpha, best_f1, best_model = None, -1, None
    for alpha in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
        nb = NaiveBayesClassifier(alpha=alpha).fit(Xtr, ytr)
        dev_pred = nb.predict(Xdev)
        f1 = evaluate(ydev, dev_pred)["f1"]
        if f1 > best_f1:
            best_f1, best_alpha, best_model = f1, alpha, nb
    test_pred = best_model.predict(Xtest)
    dev_pred = best_model.predict(Xdev)
    tuned["naive_bayes"] = {
        "tokenization": tok_name,
        "best_params": {"alpha": best_alpha},
        "dev_before": table1["naive_bayes"][tok_name]["dev"],
        "dev_after": evaluate(ydev, dev_pred),
        "test_after": evaluate(ytest, test_pred),
    }

    # ---------------- Bigram LM: tune smoothing alpha ----------------
    tok_name = best_tokenization(table1, "bigram_model")
    print(f"Bigram LM: best tokenization from Phase 1 = {tok_name}; tuning smoothing alpha...")
    Xtr = tok_all(train_df, tok_name)
    Xdev = tok_all(dev_df, tok_name)
    Xtest = tok_all(test_df, tok_name)
    best_alpha, best_f1, best_model = None, -1, None
    for alpha in [0.01, 0.1, 0.5, 1.0, 2.0]:
        blm = BigramLMClassifier(alpha=alpha).fit(Xtr, ytr)
        dev_pred = blm.predict(Xdev)
        f1 = evaluate(ydev, dev_pred)["f1"]
        if f1 > best_f1:
            best_f1, best_alpha, best_model = f1, alpha, blm
    test_pred = best_model.predict(Xtest)
    dev_pred = best_model.predict(Xdev)
    tuned["bigram_model"] = {
        "tokenization": tok_name,
        "best_params": {"alpha": best_alpha},
        "dev_before": table1["bigram_model"][tok_name]["dev"],
        "dev_after": evaluate(ydev, dev_pred),
        "test_after": evaluate(ytest, test_pred),
    }

    # ---------------- SVM: tune C / kernel / gamma ----------------
    tok_name = best_tokenization(table1, "svm")
    print(f"SVM: best tokenization from Phase 1 = {tok_name}; tuning C/kernel/gamma...")
    use_text2vec = (tok_name == "text2vec")
    if use_text2vec:
        w2v = AveragedWord2VecVectorizer(vector_size=100)
        Xtr_v = w2v.fit_transform(tok_all(train_df, "jieba"))
        Xdev_v = w2v.transform(tok_all(dev_df, "jieba"))
        Xtest_v = w2v.transform(tok_all(test_df, "jieba"))
    else:
        Xtr_v = tok_all(train_df, tok_name)
        Xdev_v = tok_all(dev_df, tok_name)
        Xtest_v = tok_all(test_df, tok_name)

    best_params, best_f1, best_preds = None, -1, None
    for C in [0.1, 1.0, 10.0, 100.0]:
        for kernel in ["rbf", "linear"]:
            vec, clf = build_svm(C=C, kernel=kernel)
            if use_text2vec:
                train_pred, dev_pred, test_pred = fit_predict_dense(clf, Xtr_v, ytr, Xdev_v, Xtest_v)
            else:
                train_pred, dev_pred, test_pred = fit_predict(vec, clf, Xtr_v, ytr, Xdev_v, Xtest_v)
            f1 = evaluate(ydev, dev_pred)["f1"]
            if f1 > best_f1:
                best_f1, best_params, best_preds = f1, {"C": C, "kernel": kernel}, (dev_pred, test_pred)
    tuned["svm"] = {
        "tokenization": tok_name,
        "best_params": best_params,
        "dev_before": table1["svm"][tok_name]["dev"],
        "dev_after": evaluate(ydev, best_preds[0]),
        "test_after": evaluate(ytest, best_preds[1]),
    }

    # ---------------- Logistic Regression: broader C sweep + tokenization already dev-picked ----------------
    tok_name = best_tokenization(table1, "logistic_regression")
    print(f"Logistic Regression: best tokenization from Phase 1 = {tok_name}; broader C sweep...")
    use_text2vec = (tok_name == "text2vec")
    if use_text2vec:
        w2v = AveragedWord2VecVectorizer(vector_size=100)
        Xtr_v = w2v.fit_transform(tok_all(train_df, "jieba"))
        Xdev_v = w2v.transform(tok_all(dev_df, "jieba"))
        Xtest_v = w2v.transform(tok_all(test_df, "jieba"))
    else:
        Xtr_v = tok_all(train_df, tok_name)
        Xdev_v = tok_all(dev_df, tok_name)
        Xtest_v = tok_all(test_df, tok_name)

    best_C, best_f1, best_preds = None, -1, None
    for C in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        vec, clf = build_logistic_regression(C=C)
        if use_text2vec:
            train_pred, dev_pred, test_pred = fit_predict_dense(clf, Xtr_v, ytr, Xdev_v, Xtest_v)
        else:
            train_pred, dev_pred, test_pred = fit_predict(vec, clf, Xtr_v, ytr, Xdev_v, Xtest_v)
        f1 = evaluate(ydev, dev_pred)["f1"]
        if f1 > best_f1:
            best_f1, best_C, best_preds = f1, C, (dev_pred, test_pred)
    tuned["logistic_regression"] = {
        "tokenization": tok_name,
        "best_params": {"C": best_C},
        "dev_before": table1["logistic_regression"][tok_name]["dev"],
        "dev_after": evaluate(ydev, best_preds[0]),
        "test_after": evaluate(ytest, best_preds[1]),
    }

    # ---------------- LSTM: tune embedding_dim / lstm_units / dropout ----------------
    tok_name = best_tokenization(table1, "lstm")
    print(f"LSTM: best tokenization from Phase 1 = {tok_name}; tuning architecture (this is slow)...")
    Xtr = tok_all(train_df, tok_name)
    Xdev = tok_all(dev_df, tok_name)
    Xtest = tok_all(test_df, tok_name)

    configs = [
        {"embedding_dim": 128, "lstm_units": 32, "dropout": 0.0},
        {"embedding_dim": 128, "lstm_units": 64, "dropout": 0.3},
        {"embedding_dim": 256, "lstm_units": 64, "dropout": 0.3},
    ]
    best_params, best_f1, best_preds = None, -1, None
    for cfg in configs:
        train_pred, dev_pred, test_pred = lstm_train_predict(
            Xtr, ytr, Xdev, ydev, test_tokens=Xtest, epochs=8, **cfg
        )
        f1 = evaluate(ydev, dev_pred)["f1"]
        if f1 > best_f1:
            best_f1, best_params, best_preds = f1, cfg, (dev_pred, test_pred)
    tuned["lstm"] = {
        "tokenization": tok_name,
        "best_params": best_params,
        "dev_before": table1["lstm"][tok_name]["dev"],
        "dev_after": evaluate(ydev, best_preds[0]),
        "test_after": evaluate(ytest, best_preds[1]),
    }

    with open("results/tuned_results.json", "w") as f:
        json.dump(tuned, f, indent=2, ensure_ascii=False)
    print(json.dumps(tuned, indent=2, ensure_ascii=False))
    print(f"\nWritten to results/tuned_results.json")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
