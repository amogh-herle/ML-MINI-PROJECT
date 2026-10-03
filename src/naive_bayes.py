"""Naive Bayes classifier, implemented directly from the paper's equations
(Sec 4.1, eq. 1-3), rather than relying on sklearn's MultinomialNB, so the
math matches exactly.

phi_{k|y=1} = ( sum_i sum_j 1{x_j^(i)=k AND y^(i)=1} + 1 ) / ( sum_i 1{y^(i)=1} n_i + |V| )
phi_{k|y=0} = ( sum_i sum_j 1{x_j^(i)=k AND y^(i)=0} + 1 ) / ( sum_i 1{y^(i)=0} n_i + |V| )

i.e. Laplace(add-1)-smoothed multinomial Naive Bayes over token counts.

Linear interpolation (eq. 3) combines bigram- and trigram-tokenized log
probabilities:
    log P(y=1|x) = lambda * log P_bigram(y=1|x) + (1-lambda) * log P_trigram(y=1|x)
"""
import math
from collections import Counter, defaultdict


class NaiveBayesClassifier:
    """Multinomial Naive Bayes with Laplace (add-1) smoothing, eq. 1-2."""

    def __init__(self):
        self.vocab = set()
        self.token_counts = {0: Counter(), 1: Counter()}
        self.class_token_totals = {0: 0, 1: 0}
        self.class_doc_counts = {0: 0, 1: 0}
        self.n_docs = 0

    def fit(self, token_lists, labels):
        for tokens, y in zip(token_lists, labels):
            self.class_doc_counts[y] += 1
            self.n_docs += 1
            for tok in tokens:
                self.vocab.add(tok)
                self.token_counts[y][tok] += 1
                self.class_token_totals[y] += 1
        return self

    def _log_phi(self, token, y):
        V = len(self.vocab)
        return math.log(
            (self.token_counts[y][token] + 1) / (self.class_token_totals[y] + V)
        )

    def log_prob_y1(self, tokens):
        """Return (log P(x|y=1) + log P(y=1)) - (log P(x|y=0) + log P(y=0))
        style log-odds score; positive => predict 1."""
        log_prior_1 = math.log(self.class_doc_counts[1] / self.n_docs)
        log_prior_0 = math.log(self.class_doc_counts[0] / self.n_docs)
        log_p1 = log_prior_1 + sum(self._log_phi(t, 1) for t in tokens)
        log_p0 = log_prior_0 + sum(self._log_phi(t, 0) for t in tokens)
        return log_p1, log_p0

    def predict_one(self, tokens):
        log_p1, log_p0 = self.log_prob_y1(tokens)
        return 1 if log_p1 >= log_p0 else 0

    def predict(self, token_lists):
        return [self.predict_one(t) for t in token_lists]


class InterpolatedNaiveBayes:
    """Linear interpolation of a bigram-tokenized and trigram-tokenized
    Naive Bayes model, eq. 3: log P(y=1|x) = lambda*log P_bigram + (1-lambda)*log P_trigram.
    Prediction compares the interpolated log P(y=1|x) against interpolated log P(y=0|x).
    """

    def __init__(self, lam=0.8):
        self.lam = lam
        self.nb_bigram = NaiveBayesClassifier()
        self.nb_trigram = NaiveBayesClassifier()

    def fit(self, bigram_token_lists, trigram_token_lists, labels):
        self.nb_bigram.fit(bigram_token_lists, labels)
        self.nb_trigram.fit(trigram_token_lists, labels)
        return self

    def predict(self, bigram_token_lists, trigram_token_lists):
        preds = []
        for bt, tt in zip(bigram_token_lists, trigram_token_lists):
            b1, b0 = self.nb_bigram.log_prob_y1(bt)
            t1, t0 = self.nb_trigram.log_prob_y1(tt)
            score1 = self.lam * b1 + (1 - self.lam) * t1
            score0 = self.lam * b0 + (1 - self.lam) * t0
            preds.append(1 if score1 >= score0 else 0)
        return preds
