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
from sklearn.svm import SVC
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


def build_svm(C=1.0, kernel="rbf", gamma="scale"):
    """Paper (Sec 4.2) uses scikit-learn's SVM with no hyperparameter search
    mentioned -> default SVC (kernel='rbf', C=1.0, gamma='scale'). Phase-2
    tuning sweeps C/kernel/gamma on dev."""
    vectorizer = CountVectorizer(
        tokenizer=identity_tokenizer, preprocessor=lambda x: x, lowercase=False
    )
    clf = SVC(kernel=kernel, C=C, gamma=gamma)
    return vectorizer, clf


def fit_predict(vectorizer, clf, X_train, y_train, X_dev, X_test=None):
    Xtr = vectorizer.fit_transform(X_train)
    clf.fit(Xtr, y_train)
    dev_pred = clf.predict(vectorizer.transform(X_dev))
    test_pred = clf.predict(vectorizer.transform(X_test)) if X_test is not None else None
    train_pred = clf.predict(Xtr)
    return train_pred, dev_pred, test_pred


def fit_predict_dense(clf, Xtr, y_train, Xdev, Xtest=None):
    """Same as fit_predict but for already-vectorized (e.g. text2vec/word2vec)
    dense feature matrices, where fitting a sklearn vectorizer doesn't apply."""
    clf.fit(Xtr, y_train)
    train_pred = clf.predict(Xtr)
    dev_pred = clf.predict(Xdev)
    test_pred = clf.predict(Xtest) if Xtest is not None else None
    return train_pred, dev_pred, test_pred
