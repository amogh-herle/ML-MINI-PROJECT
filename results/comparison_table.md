| Classifier | Tokenization | Train Acc | Dev Acc | Dev Precision | Dev Recall | Dev F1 |
|---|---|---|---|---|---|---|
| naive_bayes | unigram | 0.8124 | 0.6400 | 0.6684 | 0.5950 | 0.6295 |
| naive_bayes | bigram | 0.9776 | 0.6329 | 0.6582 | 0.5950 | 0.6250 |
| naive_bayes | trigram | 0.9881 | 0.6306 | 0.6311 | 0.6773 | 0.6534 |
| naive_bayes | jieba | 0.9429 | 0.6412 | 0.6675 | 0.6018 | 0.6330 |
| naive_bayes | bigram_trigram_interpolated | 0.9881 | 0.6588 | 0.6705 | 0.6613 | 0.6659 |
| bigram_model | unigram | 0.9814 | 0.6318 | 0.6384 | 0.6545 | 0.6463 |
| bigram_model | jieba | 0.9876 | 0.5906 | 0.6005 | 0.6087 | 0.6045 |
| svm | unigram | 0.8352 | 0.6471 | 0.6252 | 0.7826 | 0.6951 |
| svm | bigram | 0.8871 | 0.6012 | 0.5768 | 0.8421 | 0.6847 |
| svm | trigram | 0.8548 | 0.5353 | 0.5269 | 0.9405 | 0.6754 |
| svm | jieba | 0.7962 | 0.6129 | 0.5877 | 0.8284 | 0.6876 |
| svm | text2vec | 0.5681 | 0.5376 | 0.5296 | 0.8993 | 0.6667 |
| logistic_regression | unigram | 0.7424 | 0.6600 | 0.6321 | 0.8101 | 0.7101 |
| logistic_regression | bigram | 0.7419 | 0.5847 | 0.5588 | 0.9130 | 0.6933 |
| logistic_regression | trigram | 0.6448 | 0.5482 | 0.5348 | 0.9314 | 0.6795 |
| logistic_regression | jieba | 0.7267 | 0.6024 | 0.5838 | 0.7895 | 0.6712 |
| logistic_regression | text2vec | 0.6105 | 0.5671 | 0.5631 | 0.7048 | 0.6260 |
| lstm | unigram | 0.9738 | 0.6188 | 0.6101 | 0.7162 | 0.6589 |
| lstm | bigram | 0.9919 | 0.5612 | 0.6053 | 0.4211 | 0.4966 |
| lstm | trigram | 0.9500 | 0.5035 | 0.5153 | 0.5789 | 0.5453 |
| lstm | jieba | 0.9900 | 0.5976 | 0.6004 | 0.6499 | 0.6242 |

### Table 2 — Best 2 models (by dev F1), test-set performance
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| logistic_regression_unigram | 0.6553 | 0.6321 | 0.7922 | 0.7031 |
| svm_unigram | 0.6647 | 0.6435 | 0.7831 | 0.7065 |

### Phase 2 — Before/after tuning (dev-only selection, test touched once)
| Classifier | Tokenization | Best Params | Dev F1 Before | Dev F1 After | Test F1 After |
|---|---|---|---|---|---|
| naive_bayes | trigram | alpha=0.1 | 0.6534 | 0.6566 | 0.6522 |
| bigram_model | unigram | alpha=1.0 | 0.6463 | 0.6463 | 0.6426 |
| svm | unigram | C=1.0, kernel=rbf | 0.6951 | 0.6951 | 0.7065 |
| logistic_regression | unigram | C=0.1 | 0.7101 | 0.7101 | 0.7031 |
| lstm | unigram | embedding_dim=128, lstm_units=64, dropout=0.3 | 0.6589 | 0.6569 | 0.6552 |