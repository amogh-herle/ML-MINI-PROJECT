"""LDA + Logistic Regression (paper's Experiment 1 / baseline model).

"Each decision is represented by a term frequency-inverse document frequency
(tf-idf) matrix. The gensim package was used to generate the LDA model. LR
was used to classify decisions into one of 279 issue_code or 15 issue areas."

Pipeline: tokenize -> gensim Dictionary -> TF-IDF corpus -> LDA topic model
-> per-document topic-distribution vector (dense, length num_topics) ->
sklearn LogisticRegression on top of those vectors.
"""
import numpy as np
from gensim import corpora, models
from sklearn.linear_model import LogisticRegression


class LDAFeaturizer:
    def __init__(self, num_topics=50, passes=5, seed=42):
        self.num_topics = num_topics
        self.passes = passes
        self.seed = seed
        self.dictionary = None
        self.tfidf = None
        self.lda = None

    def fit(self, tokenized_docs):
        self.dictionary = corpora.Dictionary(tokenized_docs)
        self.dictionary.filter_extremes(no_below=3, no_above=0.5)
        bow_corpus = [self.dictionary.doc2bow(d) for d in tokenized_docs]
        self.tfidf = models.TfidfModel(bow_corpus)
        tfidf_corpus = self.tfidf[bow_corpus]
        self.lda = models.LdaModel(
            tfidf_corpus, num_topics=self.num_topics, id2word=self.dictionary,
            passes=self.passes, random_state=self.seed,
        )
        return self

    def _doc_vector(self, tokens):
        bow = self.dictionary.doc2bow(tokens)
        tfidf_vec = self.tfidf[bow]
        topic_dist = self.lda.get_document_topics(tfidf_vec, minimum_probability=0.0)
        vec = np.zeros(self.num_topics)
        for topic_id, prob in topic_dist:
            vec[topic_id] = prob
        return vec

    def transform(self, tokenized_docs):
        return np.vstack([self._doc_vector(d) for d in tokenized_docs])

    def fit_transform(self, tokenized_docs):
        self.fit(tokenized_docs)
        return self.transform(tokenized_docs)


def train_lda_lr(X_train_vecs, y_train, C=1.0, max_iter=1000):
    clf = LogisticRegression(C=C, max_iter=max_iter)
    clf.fit(X_train_vecs, y_train)
    return clf
