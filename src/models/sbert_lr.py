"""Phase 5: pretrained Sentence-BERT embeddings + Logistic Regression.

Doc2Vec (Phases 1-4) is a 2014-era model trained from scratch on our ~5,600
training documents -- far too little data for an embedding model to learn
good general-purpose representations, which is the main reason its accuracy
tops out around 60-69%. SBERT instead starts from a model already pretrained
on ~1B+ sentence pairs; we only use it to *encode* text (no training), so it
already "knows" language before it ever sees a single court opinion.

Same interface as Doc2VecFeaturizer (fit_transform/transform) so it drops
into the exact same comparison scripts.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

MODEL_NAME = "all-MiniLM-L6-v2"  # ~5x smaller/faster than all-mpnet-base-v2 on CPU, still pretrained on 1B+ pairs


class SBERTFeaturizer:
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def fit(self, texts):
        return self  # pretrained: nothing to fit

    def transform(self, texts):
        """texts: raw document strings (pipeline.load_split_tokenize()'s
        'train_text'/'dev_text'/'test_text' -- NOT the stopword-stripped
        'train_tok' used by LDA/Doc2Vec; SBERT needs real sentences)."""
        return np.asarray(self.model.encode(texts, show_progress_bar=False, batch_size=32))

    def fit_transform(self, texts):
        return self.transform(texts)


def train_sbert_lr(X_train_vecs, y_train, C=1.0, max_iter=1000):
    clf = LogisticRegression(C=C, max_iter=max_iter)
    clf.fit(X_train_vecs, y_train)
    return clf
