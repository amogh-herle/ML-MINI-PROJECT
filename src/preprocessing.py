"""Load and preprocess the Supreme Court Database corpus (via textacy), matching
the reference paper's "Classification of Legal Text" (Iyer, CS229 2020) dataset:
~8,400 US Supreme Court opinions (1946-2016), each labelled with:
  - issue_area: 15 broad human-curated categories (paper's "15-Labels")
  - issue: fine-grained issue code, ~260-280 distinct values (paper's "279 Labels")

Deviation from the paper: legal opinions run very long (median ~34k characters
here); we truncate each opinion to its first MAX_CHARS characters before
vectorization, for tractable LDA/Doc2Vec training time. This keeps the
dominant vocabulary (most legal opinions state the issue and holding early)
while staying within a reasonable compute budget.
"""
import re
import textacy.datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

MAX_CHARS = 4000
SEED = 42
DATA_DIR = "data/textacy_scotus"

WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = WS_RE.sub(" ", text).strip()
    return text[:MAX_CHARS]


def load_corpus(max_docs=None):
    """Returns list of dicts: {text, issue_area, issue}. Drops records with
    missing/unknown issue_area (-1) or missing issue code, matching the
    paper's use of only human-curated-category records."""
    ds = textacy.datasets.SupremeCourt(data_dir=DATA_DIR)
    ds.download()  # no-op if already downloaded
    records = []
    for text, meta in ds.records(limit=max_docs):
        issue_area = meta.get("issue_area")
        issue = meta.get("issue")
        if issue_area is None or issue_area == -1 or not issue:
            continue
        records.append({
            "text": clean_text(text),
            "issue_area": issue_area,
            "issue": issue,
        })
    return records


def split_records(records, train_size=0.7, dev_size=0.15):
    """70/15/15 train/dev/test split, stratified by issue_area (the coarser,
    better-balanced label)."""
    labels = [r["issue_area"] for r in records]
    train, rest = train_test_split(
        records, train_size=train_size, random_state=SEED, stratify=labels
    )
    rest_labels = [r["issue_area"] for r in rest]
    dev_frac = dev_size / (1 - train_size)
    dev, test = train_test_split(
        rest, train_size=dev_frac, random_state=SEED, stratify=rest_labels
    )
    return train, dev, test


def simple_tokenize(text: str):
    toks = re.findall(r"[a-zA-Z]{2,}", text.lower())
    return [t for t in toks if t not in ENGLISH_STOP_WORDS]
