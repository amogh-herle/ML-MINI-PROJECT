"""LSTM classifier, Sec 4.5, eq. 7-13.

Architecture exactly as described in the paper:
  - Embedding layer, dimension 256
  - One LSTM layer, 64 units
  - Dense layer, sigmoid activation
  - Adam optimizer: lr=0.0005, beta_1=0.9, beta_2=0.999
  - Batch size 16
  - Binary cross-entropy loss (eq. 13)
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras


def build_vocab(token_lists, max_vocab=20000):
    from collections import Counter
    counts = Counter()
    for toks in token_lists:
        counts.update(toks)
    vocab = ["<pad>", "<unk>"] + [w for w, _ in counts.most_common(max_vocab - 2)]
    return {w: i for i, w in enumerate(vocab)}


def encode(token_lists, word2idx, max_len):
    seqs = []
    for toks in token_lists:
        ids = [word2idx.get(t, 1) for t in toks][:max_len]
        ids = ids + [0] * (max_len - len(ids))
        seqs.append(ids)
    return np.array(seqs, dtype=np.int32)


def build_lstm_model(vocab_size, max_len, embedding_dim=256, lstm_units=64,
                      dropout=0.0, learning_rate=0.0005):
    layers = [
        keras.layers.Input(shape=(max_len,)),
        keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True),
        keras.layers.LSTM(lstm_units, dropout=dropout),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
    model = keras.Sequential(layers)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_and_predict(train_tokens, y_train, dev_tokens, y_dev, test_tokens=None,
                       max_len=60, epochs=8, batch_size=16, seed=42,
                       embedding_dim=256, lstm_units=64, dropout=0.0, learning_rate=0.0005):
    tf.random.set_seed(seed)
    word2idx = build_vocab(train_tokens)
    vocab_size = len(word2idx)

    Xtr = encode(train_tokens, word2idx, max_len)
    Xdev = encode(dev_tokens, word2idx, max_len)
    ytr = np.array(y_train, dtype=np.float32)
    ydev = np.array(y_dev, dtype=np.float32)

    model = build_lstm_model(vocab_size, max_len, embedding_dim=embedding_dim,
                              lstm_units=lstm_units, dropout=dropout, learning_rate=learning_rate)
    model.fit(
        Xtr, ytr,
        validation_data=(Xdev, ydev),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    train_pred = (model.predict(Xtr, verbose=0).ravel() >= 0.5).astype(int)
    dev_pred = (model.predict(Xdev, verbose=0).ravel() >= 0.5).astype(int)
    test_pred = None
    if test_tokens is not None:
        Xtest = encode(test_tokens, word2idx, max_len)
        test_pred = (model.predict(Xtest, verbose=0).ravel() >= 0.5).astype(int)
    return train_pred, dev_pred, test_pred
