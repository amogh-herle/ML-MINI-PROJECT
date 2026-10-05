### Table 1 — Phase 1 replication: accuracy by model & label granularity
| Model | Label task | Dev Acc | Dev Macro-F1 | Test Acc | Test Macro-F1 |
|---|---|---|---|---|---|
| lda_lr | issue_area (15-label) | 0.3824 | 0.1557 | 0.3935 | 0.1487 |
| lda_lr | issue (fine-grained) | 0.0803 | 0.0116 | 0.0827 | 0.0129 |
| doc2vec_lr | issue_area (15-label) | 0.5757 | 0.4742 | 0.5915 | 0.4629 |
| doc2vec_lr | issue (fine-grained) | 0.3674 | 0.2992 | 0.3709 | 0.2901 |

### Table 2 — Per-category test performance on issue_area (15 classes)

**lda_lr**
| Category | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 1 | 0.38 | 0.88 | 0.53 | 277 |
| 2 | 0.35 | 0.15 | 0.21 | 198 |
| 3 | 0.38 | 0.05 | 0.09 | 94 |
| 4 | 0.00 | 0.00 | 0.00 | 51 |
| 5 | 0.00 | 0.00 | 0.00 | 16 |
| 6 | 0.00 | 0.00 | 0.00 | 13 |
| 7 | 0.71 | 0.23 | 0.34 | 44 |
| 8 | 0.40 | 0.70 | 0.51 | 242 |
| 9 | 0.39 | 0.07 | 0.12 | 157 |
| 10 | 0.00 | 0.00 | 0.00 | 50 |
| 11 | 0.00 | 0.00 | 0.00 | 8 |
| 12 | 1.00 | 0.07 | 0.12 | 45 |
| 13 | 0.00 | 0.00 | 0.00 | 2 |

**doc2vec_lr**
| Category | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 1 | 0.82 | 0.73 | 0.77 | 277 |
| 2 | 0.61 | 0.57 | 0.59 | 198 |
| 3 | 0.62 | 0.66 | 0.64 | 94 |
| 4 | 0.30 | 0.45 | 0.36 | 51 |
| 5 | 0.20 | 0.38 | 0.26 | 16 |
| 6 | 0.22 | 0.54 | 0.31 | 13 |
| 7 | 0.52 | 0.64 | 0.57 | 44 |
| 8 | 0.70 | 0.59 | 0.64 | 242 |
| 9 | 0.48 | 0.44 | 0.46 | 157 |
| 10 | 0.39 | 0.36 | 0.38 | 50 |
| 11 | 0.21 | 0.62 | 0.31 | 8 |
| 12 | 0.73 | 0.73 | 0.73 | 45 |
| 13 | 0.00 | 0.00 | 0.00 | 2 |

### Phase 2 — Before/after tuning (dev-selected, test touched once)
| Model+Task | Extra Params | Best C | Dev Macro-F1 (after) | Test Acc (after) | Test Macro-F1 (after) |
|---|---|---|---|---|---|
| lda_lr_issue_area | num_topics=100 | 10.0 | 0.3055 | 0.4871 | 0.2674 |
| lda_lr_issue | num_topics=100 | 10.0 | 0.0840 | 0.1988 | 0.0843 |
| doc2vec_lr_issue_area | vector_size=50 | 1.0 | 0.5220 | 0.6224 | 0.4769 |
| doc2vec_lr_issue | vector_size=50 | 0.1 | 0.3125 | 0.4202 | 0.2888 |