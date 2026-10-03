"""Bigram language model classifier, Sec 4.4, eq. 6.

Denote text x = (x_1, ..., x_n), with b/e start/end tokens, sentiment y.
Estimate p(x_i | x_{i-1}, y) on the training set with add-1 (Laplace) smoothing.
Then classify a new text x by:
    p(y|x) ~ p(e | x_n, y) * p(x_n | x_{n-1}, y) * ... * p(x_2 | x_1, y) * p(x_1 | b, y) * p(y)
Predict the y that maximizes this (computed in log-space for stability).
"""
import math
from collections import Counter, defaultdict

B, E = "<b>", "<e>"


class BigramLMClassifier:
    def __init__(self):
        # bigram_counts[y][(prev, cur)] = count ; unigram_counts[y][prev] = count (for normalization)
        self.bigram_counts = {0: Counter(), 1: Counter()}
        self.context_counts = {0: Counter(), 1: Counter()}
        self.vocab = set([B, E])
        self.class_doc_counts = {0: 0, 1: 0}
        self.n_docs = 0

    def fit(self, token_lists, labels):
        for tokens, y in zip(token_lists, labels):
            self.class_doc_counts[y] += 1
            self.n_docs += 1
            seq = [B] + list(tokens) + [E]
            for tok in tokens:
                self.vocab.add(tok)
            for prev, cur in zip(seq[:-1], seq[1:]):
                self.bigram_counts[y][(prev, cur)] += 1
                self.context_counts[y][prev] += 1
        return self

    def _log_p_next(self, prev, cur, y):
        V = len(self.vocab)
        return math.log(
            (self.bigram_counts[y][(prev, cur)] + 1) / (self.context_counts[y][prev] + V)
        )

    def _score(self, tokens, y):
        log_prior = math.log(self.class_doc_counts[y] / self.n_docs)
        seq = [B] + list(tokens) + [E]
        score = log_prior
        for prev, cur in zip(seq[:-1], seq[1:]):
            score += self._log_p_next(prev, cur, y)
        return score

    def predict_one(self, tokens):
        s1 = self._score(tokens, 1)
        s0 = self._score(tokens, 0)
        return 1 if s1 >= s0 else 0

    def predict(self, token_lists):
        return [self.predict_one(t) for t in token_lists]
