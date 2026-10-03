"""Naive Bayes and Logistic Regression classifiers, replicating the paper's
baseline (Sec 4.1) and best classical model (Sec 4.3), using scikit-learn.

Paper used: Naive Bayes w/ Laplace smoothing (baseline), Logistic Regression
with raw-count / text2vec features (C=2 for jieba tokenization). We substitute
TF-IDF for text2vec (text2vec word-embeddings are out of scope for the time
budget here) and note this deviation in the README.
"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def identity_tokenizer(tokens):
    return tokens


def build_naive_bayes(alpha=1.0):
    vectorizer = CountVectorizer(
        tokenizer=identity_tokenizer, preprocessor=lambda x: x, lowercase=False
    )
    clf = MultinomialNB(alpha=alpha)
    return vectorizer, clf


def build_logistic_regression(C=2.0):
    vectorizer = TfidfVectorizer(
        tokenizer=identity_tokenizer, preprocessor=lambda x: x, lowercase=False
    )
    clf = LogisticRegression(C=C, max_iter=1000)
    return vectorizer, clf


def fit_predict(vectorizer, clf, X_train, y_train, X_dev, X_test=None):
    Xtr = vectorizer.fit_transform(X_train)
    clf.fit(Xtr, y_train)
    dev_pred = clf.predict(vectorizer.transform(X_dev))
    test_pred = clf.predict(vectorizer.transform(X_test)) if X_test is not None else None
    train_pred = clf.predict(Xtr)
    return train_pred, dev_pred, test_pred
