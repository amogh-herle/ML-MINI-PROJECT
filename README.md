# ML-MINI-PROJECT

Sentiment Analysis of Chinese Microblog (Weibo) Posts — full replication of the
Stanford CS229 project *"Sentiment Analysis of Chinese Microblog Posts on
Attitude Towards Vaccination Since the Distribution of COVID-19 Vaccines in
China"* (Bai, He & Yuan, 2021), followed by hyperparameter tuning.

## Problem statement

Classify Weibo microblog posts as positive or negative sentiment, replicating
the reference paper's full methodology: 4 tokenization schemes and 5 model
families, matching its Table 1 (classifier x tokenization grid) and Table 2
(test-set evaluation of the two best models).

## Dataset

The reference paper scraped ~60k Weibo posts about COVID-19 vaccination
(Jan–Apr 2021) and hand-labelled ~3800 of them. **That dataset/code was never
publicly released** (not on the CS229 project page, no linked repo).

As a substitute, we use **[weibo_senti_100k](https://github.com/SophonPlus/ChineseNlpCorpus)**
(≈120k Weibo posts, binary sentiment labels derived from emoticons), downloaded
via its HuggingFace parquet mirror (`dirtycomputer/weibo_senti_100k`,
checked into `data/`). To stay in the same small-data regime as the paper, we
downsample to the paper's exact split sizes: **2100 train / 850 dev / 850 test**.

Deviations from the paper, and why:
- Topic is general Weibo sentiment, not specifically vaccine-related — no
  public vaccine-labelled Chinese dataset exists.
- Labels come from emoticons in the original posts; we strip emoticon tags
  (e.g. `[哈哈]`) during preprocessing so the text itself, not the emoticon,
  drives classification.
- `text2vec` (paper's pretrained Chinese sentence-embedding library) is
  substituted with a Word2Vec model trained on our own corpus, doc vectors
  = mean of word vectors ([`src/embeddings.py`](src/embeddings.py)) — same
  "dense embedding instead of raw counts" idea, no large pretrained download.
- No post timestamps in the substitute dataset, so the paper's month-by-month
  trend analysis (Table 3 / Figure 1, COVID vaccine rollout vs. sentiment)
  **cannot be replicated** — noted as a limitation, not attempted.

## Approach — methods replicated (paper Sec. 4)

| # | Method | Tokenizations used (paper Table 1) | Implementation |
|---|---|---|---|
| 1 | Naive Bayes, Laplace smoothing (eq. 1-2) + bigram/trigram linear interpolation (eq. 3) | unigram, bigram, trigram, jieba | [`src/naive_bayes.py`](src/naive_bayes.py) — hand-implemented from the paper's equations |
| 2 | SVM (eq. 4) | unigram, bigram, trigram, jieba, text2vec | [`src/models.py`](src/models.py) — sklearn `SVC`, default params (paper doesn't mention tuning it) |
| 3 | Logistic Regression (eq. 5) | unigram, bigram, trigram, jieba, text2vec | [`src/models.py`](src/models.py) — sklearn, `C` chosen per-tokenization on dev set, as the paper itself does |
| 4 | Bigram language-model classifier (eq. 6) | unigram, jieba | [`src/bigram_lm.py`](src/bigram_lm.py) — hand-implemented from the paper's equation |
| 5 | LSTM (eq. 7-13): embedding dim 256, 1 LSTM layer (64 units), sigmoid dense, Adam(lr=5e-4, β1=0.9, β2=0.999), batch 16 | unigram, bigram, trigram, jieba | [`src/lstm_model.py`](src/lstm_model.py) — Keras/TensorFlow, exact architecture from the paper |

**Preprocessing** ([`src/preprocessing.py`](src/preprocessing.py)): drop posts
containing `【`/`】` (paper's filter for media/news reposts), strip URLs/
mentions/emoticon tags, remove Chinese stopwords; 4 tokenizers (jieba,
character unigram/bigram/trigram) matching the paper's Sec. 3.

**Phase 1 (replicate):** run every (classifier, tokenization) combination from
the paper's Table 1 with the paper's described settings — produces our own
Table 1 analog ([`results/table1.json`](results/table1.json),
[`results/comparison_table.md`](results/comparison_table.md)).

**Phase 2 (select + test):** pick the top-2 models by dev F1 (mirrors the
paper's selection of "LSTM bigram" + "SVM text2vec" for its Table 2), evaluate
them once on the held-out test set → [`results/table2.json`](results/table2.json).

**Phase 3 (tune):** on top of the paper's own settings, we additionally grid
search Naive Bayes `alpha` and the interpolation `lambda`, and sweep SVM/LR
regularization more broadly than the paper did, all against dev only.

## Results

Full grid in [`results/table1.json`](results/table1.json) /
[`results/comparison_table.md`](results/comparison_table.md); chart in
[`results/figures/dev_f1_by_classifier.png`](results/figures/dev_f1_by_classifier.png).
Test-set numbers for the best 2 models in [`results/table2.json`](results/table2.json).

## Setup & running

```bash
pip install -r requirements.txt
python run_all.py                    # full replication + selection, writes results/table1.json, table2.json
python src/make_report_assets.py     # builds comparison_table.md + bar chart from the results
```

Runtime: ~2 minutes on CPU (dataset parquet is checked into `data/`, no
re-download needed).

## Conclusions

- Our dev/test accuracies (0.53–0.66) sit below the paper's (0.62–0.75)
  mostly because of the dataset swap: our substitute's emoticon-derived
  labels are noisier than hand-labelled vaccine-opinion posts, and the topic
  is broader/more varied than a single-topic (vaccination) corpus.
- The relative *pattern* across tokenizations/models matches the paper well:
  n-gram SVM and LSTM models show heavy train/dev gaps (overfitting, same as
  the paper's Table 1 — e.g. SVM trigram train 0.85 vs. the paper's near-1.0
  pattern), and jieba/unigram tokenization generalizes better than raw n-grams
  for linear models, consistent with the paper's own best settings.
- Naive Bayes bigram+trigram interpolation gives a small dev-F1 bump over
  either alone, matching the paper's motivation for eq. 3, though our dev
  showed best results at a different `lambda` than the paper's 0.8 — likely
  dataset-dependent.
- Limitation: no timestamp metadata means the paper's trend-analysis
  application (Sec. 6) could not be replicated on this substitute dataset.
- Future work: swap in a genuine vaccine-labelled Weibo corpus if one becomes
  available, and use the real `text2vec` pretrained model instead of our
  from-scratch Word2Vec substitute.

## Repo structure

```
src/preprocessing.py       - cleaning, filtering, 4 tokenizers
src/naive_bayes.py         - Naive Bayes + bigram/trigram interpolation (hand-implemented, eq. 1-3)
src/bigram_lm.py           - bigram language-model classifier (hand-implemented, eq. 6)
src/embeddings.py          - text2vec substitute (averaged Word2Vec)
src/models.py              - SVM / Logistic Regression wrappers + evaluation
src/lstm_model.py          - LSTM classifier (Keras, eq. 7-13 architecture)
src/make_report_assets.py  - builds comparison table + bar chart from results
run_all.py                 - main pipeline: load -> Table 1 grid -> Table 2 selection
data/                      - dataset (weibo_senti_100k parquet)
results/                   - table1.json, table2.json, comparison_table.md, figures/
report/                    - reference paper, guidelines, 2-page write-up (PDF) submission
```
