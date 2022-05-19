# Even the Simplest Baseline Needs Careful Re-investigation: A Case Study on XML-CNN

This is the code for the NAACL 2022 paper "[Even the Simplest Baseline Needs Careful Re-investigation: A Case Study on XML-CNN](https://arxiv.org/abs/xxxx.xxxxx)". The repository contains an older version of [LibMultiLabel](https://github.com/ASUS-AICS/LibMultiLabel).
Please feel free to contact [Si-An Chen](https://scholar.google.com/citations?hl=en&user=XtkmEncAAAAJ) if you have any questions about the code/paper.


## Dataset
All datasets used in our experiments can be downloaded from [here](https://drive.google.com/drive/folders/1Z_Xs6zr8NNOWSFfJX5R-6eStEijsZUrZ?usp=sharing).
Each dataset contains 5 files:
- `Xf.txt`: vocabulary set of the bag-of-word (BOW) features used in [Extreme Multi-Label Repository](http://manikvarma.org/downloads/XC/XMLRepository.html). We use this set to generate `train_rv.txt` and `test_rv.txt`.
- `train.txt`, `test.txt`: training set and test set obtained from [AttentionXML](https://github.com/yourh/AttentionXML).
- `train_rv.txt`, `test_rv.txt`: training set and test set with reduced vocabulary set (`Xf.txt`).
More details can be found the Appendix in our [paper].


## Experiment Result

### EUR-Lex
| Loss/Hidden layer/Max-pooling | P@1 | P@3 | P@5 | NDCG@1 | NDCG@3 | NDCG@5 | Config |
|---|---|---|---|---|---|---|---|
| CE/N/standard | 72.78 | 59.84 | 49.94 | 72.78 | 59.84 | 49.94 | [Cfg](config/EUR-Lex/) |
| BCE/N/standard | 80.93 | 66.38 | 55.34 | 80.93 | 66.38 | 55.34 | [Cfg](config/EUR-Lex/) |
| BCE/N/dynamic | 77.67 | 64.94 | 53.29 | 77.88 | 64.58 | 53.38 | [Cfg](config/EUR-Lex/) |
| BCE/Y/standard | 76.40 | 62.78 | 51.88 | 76.56 | 62.92 | 51.84 | [Cfg](config/EUR-Lex/) |
| BCE/Y/dynamic | 77.98 | 65.11 | 53.90 | 78.94 | 65.77 | 54.15 | [Cfg](config/EUR-Lex/) |

### Wiki10-31K
| Loss/Hidden layer/Max-pooling | P@1 | P@3 | P@5 | NDCG@1 | NDCG@3 | NDCG@5 | Config |
|---|---|---|---|---|---|---|---|
| CE/N/standard | 80.70 | 64.83 | 55.43 | 80.70 | 64.83 | 55.43 | [Cfg](config/Wiki10-31K/) |
| BCE/N/standard | 82.78 | 68.07 | 57.63 | 82.78 | 68.07 | 57.63 | [Cfg](config/Wiki10-31K/) |
| BCE/N/dynamic | 83.15 | 70.32 | 59.91 | 83.37 | 70.64 | 60.16 | [Cfg](config/Wiki10-31K/) |
| BCE/Y/standard | 80.89 | 67.89 | 58.17 | 81.73 | 68.82 | 58.65 | [Cfg](config/Wiki10-31K/) |
| BCE/Y/dynamic | 84.19 | 71.55 | 61.14 | 84.70 | 71.80 | 61.03 | [Cfg](config/Wiki10-31K/) |

### AmazonCat-13K
| Loss/Hidden layer/Max-pooling | P@1 | P@3 | P@5 | NDCG@1 | NDCG@3 | NDCG@5 | Config |
|---|---|---|---|---|---|---|---|
| CE/N/standard | 91.01 | 75.07 | 60.50 | 92.85 | 76.90 | 61.76 | [Cfg](config/AmazonCat-13K/) |
| BCE/N/standard | 93.31 | 78.02 | 62.93 | 93.41 | 78.11 | 62.95 | [Cfg](config/AmazonCat-13K/) |
| BCE/N/dynamic | 93.63 | 78.55 | 63.42 | 93.65 | 78.56 | 63.41 | [Cfg](config/AmazonCat-13K/) |
| BCE/Y/standard | 94.73 | 79.64 | 63.95 | 94.73 | 79.64 | 63.94 | [Cfg](config/AmazonCat-13K/) |
| BCE/Y/dynamic | 94.79 | 80.04 | 64.49 | 94.78 | 80.03 | 64.52 | [Cfg](config/AmazonCat-13K/) |

### Amazon-670K
| Loss/Hidden layer/Max-pooling | P@1 | P@3 | P@5 | NDCG@1 | NDCG@3 | NDCG@5 | Config |
|---|---|---|---|---|---|---|---|
| CE/N/standard | 27.14 | 24.70 | 22.70 | 27.23 | 24.65 | 22.70 | [Cfg](config/Amazon-670K/) |
| BCE/N/standard | 33.38 | 29.99 | 27.47 | 33.38 | 29.99 | 27.47 | [Cfg](config/Amazon-670K/) |
| BCE/N/dynamic | 34.58 | 30.89 | 28.24 | 34.61 | 30.91 | 28.25 | [Cfg](config/Amazon-670K/) |
| BCE/Y/standard | 33.62 | 30.15 | 27.62 | 33.86 | 30.27 | 27.69 | [Cfg](config/Amazon-670K/) |
| BCE/Y/dynamic | 35.53 | 31.82 | 29.03 | 35.69 | 31.89 | 29.08 | [Cfg](config/Amazon-670K/) |
