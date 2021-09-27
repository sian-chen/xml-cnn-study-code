import argparse

import pandas as pd
from nltk.tokenize import RegexpTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', required=True)
    parser.add_argument('-o', '--output_path', required=True)
    parser.add_argument('-v', '--vocab_path', required=True)
    return parser.parse_args()


def tokenize(text):
    tokenizer = RegexpTokenizer(r'\w+')
    return [t.lower() for t in tokenizer.tokenize(text) if not t.isnumeric()]


def remove_vocab(s, vocab):
    out = []
    for w in tokenize(s):
        if w in vocab:
            out.append(w)
    return ' '.join(out)


def main():
    args = parse_args()
    with open(args.vocab_path, 'r') as fp:
        vocab = set([v.strip() for v in fp.readlines()])
    df = pd.read_csv(args.input_path, sep='\t', names=['label', 'text'])
    df['text'] = df['text'].apply(lambda s: remove_vocab(s, vocab))
    df.to_csv(args.output_path, sep='\t', header=False)


if __name__ == '__main__':
    main()
