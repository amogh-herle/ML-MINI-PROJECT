"""Doc2Vec + Logistic Regression (paper's Experiment 2).

"Doc2vec or document vectors map semantic meaning from a variable length
document to a fixed size vector... The embedding vectors extracted by the
Doc2Vec model used by LR for further classification of the legal opinions."
"""
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.linear_model import LogisticRegression


class Doc2VecFeaturizer:
    def __init__(self, vector_size=100, window=8, min_count=3, epochs=20, seed=42):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.seed = seed
        self.model = None

    def fit(self, tokenized_docs):
        tagged = [TaggedDocument(words=toks, tags=[str(i)]) for i, toks in enumerate(tokenized_docs)]
        self.model = Doc2Vec(
            vector_size=self.vector_size, window=self.window, min_count=self.min_count,
            epochs=self.epochs, seed=self.seed, workers=1,
        )
        self.model.build_vocab(tagged)
        self.model.train(tagged, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        return self

    def transform(self, tokenized_docs):
        return np.vstack([
            self.model.infer_vector(toks, epochs=self.epochs) for toks in tokenized_docs
        ])

    def fit_transform(self, tokenized_docs):
        self.fit(tokenized_docs)
        # use the vectors learned during training (more stable than re-inferring)
        return np.vstack([self.model.dv[str(i)] for i in range(len(tokenized_docs))])


def train_doc2vec_lr(X_train_vecs, y_train, C=1.0, max_iter=1000):
    clf = LogisticRegression(C=C, max_iter=max_iter)
    clf.fit(X_train_vecs, y_train)
    return clf
