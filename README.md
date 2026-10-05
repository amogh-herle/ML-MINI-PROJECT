# ML-MINI-PROJECT

Classification of US Supreme Court legal opinions — replication + hyperparameter
tuning of the Stanford CS229 project *"Classification of Legal Text"*
(Krithika Iyer, 2020).

## Problem statement

Classify US Supreme Court opinions into their legal subject-matter category,
at two granularities:
- **issue_area** — 15 broad human-curated categories (e.g. Criminal
  Procedure, Civil Rights, First Amendment).
- **issue** — ~260-280 fine-grained issue codes.

Legal opinions use long, non-standard sentence structures that break typical
NLP tooling; the paper explores whether topic-model and embedding features,
fed into a simple classifier, can still recover the human-expert categories.

## Dataset

**Real dataset, no substitute needed.** The paper uses the Supreme Court
Database (Washington University, http://scdb.wustl.edu) via the `textacy`
package's bundled `SupremeCourt` dataset — the exact same source is still
downloadable today (`textacy.datasets.SupremeCourt`, ~111MB, 8,419 opinions,
1946-2016).

After dropping opinions with no `issue_area` label and issue codes with
fewer than 10 examples (needed so every class can appear in train/dev/test),
**7,971 opinions** remain, split 70/15/15 (train/dev/test), stratified by
`issue_area`.

Deviation from the paper: opinions run very long (median ~34k characters);
we truncate each to its first 4,000 characters before vectorizing, to keep
LDA/Doc2Vec training tractable. Most opinions state the issue and holding
early, so this keeps the dominant vocabulary.

## Approach — methods replicated (paper's Experiments 1 & 2)

| # | Method | Paper section | Implementation |
|---|---|---|---|
| 1 | LDA (topic model) + Logistic Regression | Experiment 1 | [`src/lda_lr.py`](src/lda_lr.py) — gensim `Dictionary` → `TfidfModel` → `LdaModel`, document = topic-probability vector, sklearn `LogisticRegression` on top |
| 2 | Doc2Vec (paragraph embeddings) + Logistic Regression | Experiment 2 | [`src/doc2vec_lr.py`](src/doc2vec_lr.py) — gensim `Doc2Vec`, document = learned paragraph vector, sklearn `LogisticRegression` on top |

**Experiment 3 (BERT) is not replicated.** The paper's own BERT attempt ran
out of memory on Google Colab (even the paid high-RAM tier) and produced no
results — there is nothing to replicate numerically, only a limitation to
note.

**Preprocessing** ([`src/preprocessing.py`](src/preprocessing.py)): load via
textacy, drop unlabeled/rare classes, truncate to 4,000 chars, lowercase +
regex word tokenization + English stopword removal.

**Phase 1 (replicate):** [`run_all.py`](run_all.py) trains LDA+LR and
Doc2Vec+LR, evaluated on both `issue_area` (15-label) and `issue`
(fine-grained) tasks, with the paper's rough default settings (LDA
`num_topics=50`, Doc2Vec `vector_size=100`, LR `C=1.0`) → produces our Table
1/2 analogs ([`results/table1.json`](results/table1.json)).

**Phase 2 (tune):** [`tune_all.py`](tune_all.py) tunes the feature
extractor's main hyperparameter (LDA `num_topics` ∈ {20, 50, 100}, Doc2Vec
`vector_size` ∈ {50, 100, 200}) by `issue_area` dev accuracy, then tunes each
downstream Logistic Regression's `C` per label task on dev, touching test
exactly once per final config → [`results/tuned_results.json`](results/tuned_results.json).

## A note on the paper's own Table 1

The paper's text says: *"The classification accuracy for the 279 issue_code
is around 0.133. When only the issue_areas (15 categories) are considered,
the classification accuracy jumps to 0.47."* — i.e. **15-label ≈ 0.47,
279-label ≈ 0.13** for LDA+LR. But the paper's **printed Table 1 has the
columns swapped** (`15-Labels = 0.13`, `279 Labels = 0.47`), which
contradicts its own text (fewer, broader classes should be *easier*, not
harder, to predict). We report our results using the paper's own text-stated
mapping as ground truth, and note this inconsistency for transparency.

## Results

### Table 1 — Phase 1 replication (paper's default settings)

| Model | Label task | Dev Acc | Dev Macro-F1 | Test Acc | Test Macro-F1 |
|---|---|---|---|---|---|
| lda_lr | issue_area (15-label) | 0.3824 | 0.1557 | 0.3935 | 0.1487 |
| lda_lr | issue (fine-grained) | 0.0803 | 0.0116 | 0.0827 | 0.0129 |
| doc2vec_lr | issue_area (15-label) | 0.5757 | 0.4742 | 0.5915 | 0.4629 |
| doc2vec_lr | issue (fine-grained) | 0.3674 | 0.2992 | 0.3709 | 0.2901 |

**Comparison with the paper** (using its text-stated mapping): our LDA+LR
15-label test accuracy (0.39) sits reasonably close to the paper's 0.47;
our fine-grained 0.08 matches its 0.13 in the same low range. Doc2Vec+LR
15-label (0.59) is close to the paper's 0.63; fine-grained (0.37) is below
its 0.48, likely due to our stricter class filtering (rare-class dropping)
and the 4,000-character truncation.

Full JSON: [`results/table1.json`](results/table1.json). Per-category
precision/recall/F1 on `issue_area` (paper's Table 2 analog) is in
[`results/comparison_table.md`](results/comparison_table.md).

![Phase 1 test accuracy](results/figures/phase1_test_accuracy.png)

### Phase 2 — Before/after hyperparameter tuning

See [`results/tuned_results.json`](results/tuned_results.json) and
[`results/comparison_table.md`](results/comparison_table.md) for the full
before/after table (generated after `tune_all.py` finishes — it's a long
run: ~3 LDA refits + ~3 Doc2Vec refits + C sweeps, around 15-20 minutes on
CPU).

![Before vs after tuning](results/figures/before_after_tuning.png)

## Setup & running

```bash
pip install -r requirements.txt
python run_all.py                    # Phase 1: downloads corpus (~111MB, cached after first run), replicates both experiments
python tune_all.py                   # Phase 2: dev-only hyperparameter tuning (slow: several LDA/Doc2Vec refits)
python src/make_report_assets.py     # builds comparison_table.md + charts from the results
```

## Conclusions

- Both the paper's and our results agree on the overall pattern: Doc2Vec
  embeddings consistently outperform LDA topic-distribution features for
  this task, and the coarser 15-category task is much easier than the
  260+-category fine-grained task — unsurprising given how thin per-class
  training data gets at that granularity.
- LDA's macro-F1 is far below its accuracy (e.g. 0.156 vs 0.382 on
  issue_area dev) — a sign it's defaulting to the majority classes
  (Criminal Procedure, Economic Activity) and failing on rare categories,
  matching the paper's own observation that large document collections
  "do not necessarily lead to better results" for some categories.
- We did not replicate BERT: the paper's own attempt failed on memory
  limits, so there is no baseline number to reproduce. Future work could
  revisit this with a long-document Transformer variant (Longformer,
  Reformer) as the paper itself suggests.

## Repo structure

```
src/preprocessing.py       - corpus loading (textacy), cleaning, tokenizing, train/dev/test split
src/lda_lr.py              - LDA topic-distribution featurizer + Logistic Regression (Experiment 1)
src/doc2vec_lr.py          - Doc2Vec paragraph-embedding featurizer + Logistic Regression (Experiment 2)
src/make_report_assets.py  - builds comparison table + charts from results
run_all.py                 - Phase 1 pipeline: load -> featurize -> train/eval both experiments, both label tasks
tune_all.py                - Phase 2 pipeline: dev-only hyperparameter tuning per feature extractor + classifier
data/                      - cached textacy SupremeCourt corpus (downloaded on first run)
results/                   - table1.json, tuned_results.json, comparison_table.md, figures/
report/                    - reference paper, guidelines, write-up
```
