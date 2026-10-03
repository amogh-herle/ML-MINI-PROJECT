# ML-MINI-PROJECT

Sentiment Analysis of Chinese Microblog (Weibo) Posts — replication + hyperparameter
tuning of the Stanford CS229 project *"Sentiment Analysis of Chinese Microblog Posts
on Attitude Towards Vaccination Since the Distribution of COVID-19 Vaccines in China"*
(Bai, He & Yuan, 2021).

## Problem statement

Classify Weibo microblog posts as positive or negative sentiment, replicating the
reference paper's methodology (preprocessing, tokenization schemes, Naive Bayes and
Logistic Regression baselines), then tune hyperparameters to improve on the paper's
(largely untuned) results.

## Dataset

The reference paper scraped ~60k Weibo posts about COVID-19 vaccination (Jan–Apr 2021)
and hand-labelled ~3800 of them. **That dataset/code was never publicly released.**

As a substitute, we use **[weibo_senti_100k](https://github.com/SophonPlus/ChineseNlpCorpus)**
(≈120k Weibo posts, binary sentiment labels derived from emoticons), downloaded via its
HuggingFace mirror (`dirtycomputer/weibo_senti_100k`). To stay in the same small-data
regime as the paper, we downsample to the paper's exact split sizes: **2100 train / 850
dev / 850 test**.

Deviations from the paper, and why:
- Topic is general Weibo sentiment, not specifically vaccine-related (no public
  vaccine-labelled dataset exists).
- Labels come from emoticons in the original tweets; we strip emoticon tags (e.g.
  `[哈哈]`) during preprocessing so the text itself, not the emoticon, drives
  classification.
- We implement 2 of the paper's 5 model families (Naive Bayes, Logistic Regression)
  given the time budget; SVM, the bigram language model, and LSTM are noted as future
  work (see paper Sec. 4.2, 4.4, 4.5 for their formulations).
- Logistic Regression uses TF-IDF features in place of the paper's `text2vec` word
  embeddings.

## Approach

1. **Preprocessing** ([`src/preprocessing.py`](src/preprocessing.py)): drop posts
   containing `【`/`】` (paper's filter for media/news reposts), strip URLs/mentions/
   emoticon tags, remove Chinese stopwords. Four tokenizers are implemented (jieba,
   unigram, bigram, trigram char n-grams), matching the paper's Sec. 3.
2. **Phase 1 — Replicate** ([`run_all.py`](run_all.py)): Naive Bayes with Laplace
   smoothing (`alpha=1.0`, paper's default) and Logistic Regression (`C=2`, paper's
   jieba setting) on bigram-tokenized text, no hyperparameter search — mirrors the
   paper's Table 1 methodology.
3. **Phase 2 — Tune**: grid search on the **dev set only** for Naive Bayes `alpha` and
   Logistic Regression `C`; test set touched once with the best config.

## Results

See [`results/results.json`](results/results.json) for full numbers (train/dev/test
accuracy, precision, recall, F1) for both untuned (phase 1) and tuned (phase 2) models.

## Setup & running

```bash
pip install -r requirements.txt
python run_all.py
```

This downloads nothing at runtime (the dataset parquet is checked into `data/`), runs
in a few seconds, and writes results to `results/results.json`.

## Conclusions

- Untuned baselines (directly copying the paper's hyperparameters) land around
  dev/test accuracy 0.63–0.66 on our substitute dataset — comparable to the dev-set
  accuracies the paper reports (0.69–0.75) given the different dataset and much
  smaller feature vocabulary from bigram tokenization.
- Tuning `alpha`/`C` on dev gives modest F1 gains for Naive Bayes but highlights the
  precision/recall tradeoff for Logistic Regression at low `C` (high recall, lower
  precision) — the same overfitting/underfitting tension the paper observed when
  comparing tokenization schemes.
- Future work: add SVM/text2vec and LSTM per the paper's Sec. 4.2/4.5, and revisit
  on a true vaccine-labelled Weibo corpus if one becomes available.

## Repo structure

```
src/preprocessing.py   - cleaning, filtering, tokenizers
src/models.py          - Naive Bayes / Logistic Regression + evaluation
run_all.py             - main pipeline (load -> replicate -> tune -> results.json)
data/                  - dataset (weibo_senti_100k parquet)
results/results.json   - phase 1 + phase 2 metrics
report/                - 2-page write-up (PDF) submission
```
