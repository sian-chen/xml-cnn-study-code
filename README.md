# Even the Simplest Baseline Needs Careful Re-investigation: A Case Study on XML-CNN

This is the code for the NAACL 2022 paper "[Even the Simplest Baseline Needs Careful Re-investigation: A Case Study on XML-CNN](https://www.csie.ntu.edu.tw/~cjlin/papers/xmlcnn/xml_cnn_study.pdf)".
The repository is used to reproduce the experimental results in our paper.
However, results may be slightly different because of the randomness and the environment.
Please feel free to contact [Si-An Chen](https://scholar.google.com/citations?hl=en&user=XtkmEncAAAAJ) if you have any questions about the code/paper.


## Datasets
All datasets used in our experiments can be downloaded from [here](https://drive.google.com/drive/folders/1Z_Xs6zr8NNOWSFfJX5R-6eStEijsZUrZ?usp=sharing).
Each dataset contains 5 files:
- `Xf.txt`: vocabulary set of the bag-of-word (BOW) features used in [Extreme Multi-Label Repository](http://manikvarma.org/downloads/XC/XMLRepository.html). We use this set to generate `train_rv.txt` and `test_rv.txt`.
- `train.txt`, `test.txt`: training set and test set obtained from [AttentionXML](https://github.com/yourh/AttentionXML).
- `train_rv.txt`, `test_rv.txt`: training set and test set with reduced vocabulary set (`Xf.txt`).
More details can be found the Appendix in our [paper].


## Explanation of directories and files
- `XML-CNN/`: The code for the SIGIR 2017 paper "Deep learning for extreme multi-label text classification" by Liu et al.
- `config/`: The config files used in our experiments.
- `libmultilabel/`: The main experiment code. Modified from an older version of [LibMultiLabel](https://github.com/ASUS-AICS/LibMultiLabel).
- `libmultilabel/networks/`: The source code of different network architectures used in our experiments.
- `main.py`: The script for training and testing.
- `search_params.py`: The script for hyperparameter tuning.


## How to run the experiments
1. Download the datasets and place them in `data/`.
2. Run the command with a specified config file (see the next section):
```
# train
python main.py --config [CONFIG_FILE]

# test
python main.py --config [CONFIG_FILE] --eval --checkpoint_path [CHECKPOINT_PATH]
```


## Experiment Result (Table 7 and Table 15)

### EUR-Lex
| Loss/Hidden layer/Max-pooling | P@1 | P@3 | P@5 | NDCG@1 | NDCG@3 | NDCG@5 | Config |
|---|---|---|---|---|---|---|---|
| CE/N/standard  | 72.78 | 59.84 | 49.94 | 72.78 | 59.84 | 49.94 | [Cfg](config/EUR-Lex/kim_cnn_v2_best.yml) |
| BCE/N/standard | 80.93 | 66.38 | 55.34 | 80.93 | 66.38 | 55.34 | [Cfg](config/EUR-Lex/kim_cnn_v2_mlce_best.yml) |
| BCE/N/dynamic  | 77.67 | 64.94 | 53.29 | 77.88 | 64.58 | 53.38 | [Cfg](config/EUR-Lex/xml_cnn_nh_np2_best.yml) |
| BCE/Y/standard | 76.40 | 62.78 | 51.88 | 76.56 | 62.92 | 51.84 | [Cfg](config/EUR-Lex/xml_cnn_np1_best.yml) |
| BCE/Y/dynamic  | 77.98 | 65.11 | 53.90 | 78.94 | 65.77 | 54.15 | [Cfg](config/EUR-Lex/xml_cnn_np2_best.yml) |

### Wiki10-31K
| Loss/Hidden layer/Max-pooling | P@1 | P@3 | P@5 | NDCG@1 | NDCG@3 | NDCG@5 | Config |
|---|---|---|---|---|---|---|---|
| CE/N/standard  | 80.70 | 64.83 | 55.43 | 80.70 | 64.83 | 55.43 | [Cfg](config/Wiki10-31K/kim_cnn_v2_best.yml) |
| BCE/N/standard | 82.78 | 68.07 | 57.63 | 82.78 | 68.07 | 57.63 | [Cfg](config/Wiki10-31K/kim_cnn_v2_mlce_best.yml) |
| BCE/N/dynamic  | 83.15 | 70.32 | 59.91 | 83.37 | 70.64 | 60.16 | [Cfg](config/Wiki10-31K/xml_cnn_nh_np8_best.yml) |
| BCE/Y/standard | 80.89 | 67.89 | 58.17 | 81.73 | 68.82 | 58.65 | [Cfg](config/Wiki10-31K/xml_cnn_np1_best.yml) |
| BCE/Y/dynamic  | 84.19 | 71.55 | 61.14 | 84.70 | 71.80 | 61.03 | [Cfg](config/Wiki10-31K/xml_cnn_np8_best.yml) |

### AmazonCat-13K
| Loss/Hidden layer/Max-pooling | P@1 | P@3 | P@5 | NDCG@1 | NDCG@3 | NDCG@5 | Config |
|---|---|---|---|---|---|---|---|
| CE/N/standard  | 91.01 | 75.07 | 60.50 | 92.85 | 76.90 | 61.76 | [Cfg](config/AmazonCat-13K/kim_cnn_v2_best.yml) |
| BCE/N/standard | 93.31 | 78.02 | 62.93 | 93.41 | 78.11 | 62.95 | [Cfg](config/AmazonCat-13K/kim_cnn_v2_mlce_best.yml) |
| BCE/N/dynamic  | 93.63 | 78.55 | 63.42 | 93.65 | 78.56 | 63.41 | [Cfg](config/AmazonCat-13K/xml_cnn_nh_np8_best.yml) |
| BCE/Y/standard | 94.73 | 79.64 | 63.95 | 94.73 | 79.64 | 63.94 | [Cfg](config/AmazonCat-13K/xml_cnn_np1_best.yml) |
| BCE/Y/dynamic  | 94.79 | 80.04 | 64.49 | 94.78 | 80.03 | 64.52 | [Cfg](config/AmazonCat-13K/xml_cnn_np8_best.yml) |

### Amazon-670K
| Loss/Hidden layer/Max-pooling | P@1 | P@3 | P@5 | NDCG@1 | NDCG@3 | NDCG@5 | Config |
|---|---|---|---|---|---|---|---|
| CE/N/standard  | 27.14 | 24.70 | 22.70 | 27.23 | 24.65 | 22.70 | [Cfg](config/Amazon-670K/kim_cnn_v2_best.yml) |
| BCE/N/standard | 33.38 | 29.99 | 27.47 | 33.38 | 29.99 | 27.47 | [Cfg](config/Amazon-670K/kim_cnn_v2_mlce_best.yml) |
| BCE/N/dynamic  | 34.58 | 30.89 | 28.24 | 34.61 | 30.91 | 28.25 | [Cfg](config/Amazon-670K/xml_cnn_nh_np2_best.yml) |
| BCE/Y/standard | 33.62 | 30.15 | 27.62 | 33.86 | 30.27 | 27.69 | [Cfg](config/Amazon-670K/xml_cnn_np1_best.yml) |
| BCE/Y/dynamic  | 35.53 | 31.82 | 29.03 | 35.69 | 31.89 | 29.08 | [Cfg](config/Amazon-670K/xml_cnn_np2_best.yml) |
