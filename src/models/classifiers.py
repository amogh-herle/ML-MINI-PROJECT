"""Extra classifiers compared on top of the two feature extractors (LDA
topic vectors, Doc2Vec document embeddings), beyond the paper's own
Logistic Regression. Chosen to cover different ML paradigms:

  - SVM            (margin-based, kernel trick -- RBF kernel)
  - Random Forest  (ensemble of decision trees, bagging)
  - Naive Bayes    (Bayes' theorem + a generative probability model)

All three take the exact same dense feature vectors (LDA topic-probability
vectors or Doc2Vec embeddings) that feed Logistic Regression -- only the
classifier on top changes, so this is an apples-to-apples comparison of
classifier choice, with the feature extraction step held fixed.
"""
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


def build_svm(C=1.0, kernel="rbf", probability=False):
    """Support Vector Machine. Maximizes the margin between classes; RBF
    kernel lets it draw non-linear decision boundaries in feature space.
    probability=True enables predict_proba (an internal 5-fold CV calibration,
    so it's slower to fit -- only turn it on when you actually need confidence
    scores, e.g. for single-document inference)."""
    return SVC(C=C, kernel=kernel, probability=probability)


def build_random_forest(n_estimators=200, max_depth=None, seed=42):
    """Ensemble of decision trees (bagging): each tree trains on a random
    bootstrap sample and a random subset of features, predictions are
    majority-voted. Reduces overfitting vs. a single decision tree."""
    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=seed, n_jobs=-1
    )


def build_naive_bayes():
    """Gaussian Naive Bayes: applies Bayes' theorem, assuming each feature
    is normally distributed within a class and features are conditionally
    independent given the class. Used (rather than Multinomial NB) because
    both LDA topic-probabilities and Doc2Vec embeddings are continuous and
    Doc2Vec's can be negative."""
    return GaussianNB()


def train_predict(clf, Xtr, ytr, Xdev, Xtest=None):
    clf.fit(Xtr, ytr)
    dev_pred = clf.predict(Xdev)
    test_pred = clf.predict(Xtest) if Xtest is not None else None
    return dev_pred, test_pred
