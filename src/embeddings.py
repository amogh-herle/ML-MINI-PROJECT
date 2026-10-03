"""text2vec-style document embeddings (Sec 4.3 footnote 4 / Table 1 'text2vec' rows).

The paper uses shibing624/text2vec, a pretrained Chinese sentence-embedding
library. We substitute a Word2Vec model trained on our own (jieba-tokenized)
training corpus, averaging word vectors per document — the same "dense
embedding instead of raw counts" idea the paper uses to feed SVM/LR, without
requiring a large pretrained download. Documented as a deviation in README.
"""
import numpy as np
from gensim.models import Word2Vec


class AveragedWord2VecVectorizer:
    def __init__(self, vector_size=100, window=5, min_count=1, epochs=20, seed=42):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.seed = seed
        self.model = None

    def fit(self, token_lists):
        self.model = Word2Vec(
            sentences=token_lists,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            seed=self.seed,
            workers=1,
        )
        return self

    def _doc_vector(self, tokens):
        vecs = [self.model.wv[t] for t in tokens if t in self.model.wv]
        if not vecs:
            return np.zeros(self.vector_size)
        return np.mean(vecs, axis=0)

    def transform(self, token_lists):
        return np.vstack([self._doc_vector(t) for t in token_lists])

    def fit_transform(self, token_lists):
        self.fit(token_lists)
        return self.transform(token_lists)
