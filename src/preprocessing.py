"""Preprocessing utilities, following the methodology of the reference paper
(Bai, He & Yuan 2021 - "Sentiment Analysis of Chinese Microblog Posts on
Attitude Towards Vaccination Since the Distribution of COVID-19 Vaccines in China").

Steps replicated from the paper:
  1. Discard posts containing the bracket characters "【" / "】" (these are
     mostly media/news reposts rather than opinion posts).
  2. Strip Weibo emoticon tags like "[哈哈]" so labels aren't a trivial leak
     (our substitute dataset derives its labels from these emoticons).
  3. Tokenize with jieba (word segmentation), and also provide unigram/
     bigram/trigram (character n-gram) tokenizers, matching the 4 tokenization
     schemes compared in the paper.
  4. Remove Chinese stopwords.
"""
import re
import jieba

EMOTICON_RE = re.compile(r"\[[^\[\]]{1,6}\]")
BRACKET_CHARS = ("【", "】")
URL_RE = re.compile(r"https?://\S+|http://t\.cn/\S+")
MENTION_RE = re.compile(r"@[\w\-]+")

# Minimal Chinese stopword list (subset of stopwords-zh, the list used in the paper)
STOPWORDS = set("""
的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着
这 那 与 及 或 对 为 以 之 其 并 被 把 让 但 却 又 还 更 已 则 即
吗 呢 吧 啊 哦 嗯 噢 哈 呀 啦 么 的话 这个 那个 他 她 它 们 我们 你们 他们
""".split())


def basic_clean(text: str) -> str:
    """Remove URLs, @mentions, and emoticon tags; drop posts containing brackets upstream."""
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    text = EMOTICON_RE.sub("", text)
    text = text.replace("﻿", "").strip()
    return text


def has_media_brackets(text: str) -> bool:
    return any(c in text for c in BRACKET_CHARS)


def tokenize_jieba(text: str):
    return [w for w in jieba.lcut(text) if w.strip() and w not in STOPWORDS]


def tokenize_unigram(text: str):
    return [c for c in text if c.strip() and c not in STOPWORDS]


def tokenize_ngram(text: str, n: int):
    chars = [c for c in text if c.strip()]
    if len(chars) < n:
        return ["".join(chars)] if chars else []
    return ["".join(chars[i:i + n]) for i in range(len(chars) - n + 1)]


TOKENIZERS = {
    "jieba": tokenize_jieba,
    "unigram": tokenize_unigram,
    "bigram": lambda t: tokenize_ngram(t, 2),
    "trigram": lambda t: tokenize_ngram(t, 3),
}


def preprocess_dataframe(df, text_col="review", label_col="label"):
    """Apply paper's filtering: drop bracketed/media posts, clean emoticons/urls."""
    df = df.copy()
    df = df[~df[text_col].apply(has_media_brackets)]
    df[text_col] = df[text_col].apply(basic_clean)
    df = df[df[text_col].str.len() > 0]
    return df.reset_index(drop=True)
