import argparse
import os
import re
import pandas as pd
import time


def remove_vocab_xf(text, vocabs):
    """Remove word not in vocabulary.

    Args:
        text (str): Text data.
        vocabs (set): Vocabulary set.

    Returns:
        str: Text with words in the vocabulary.
    """
    text = [t for t in text.split() if t in vocabs]
    return ' '.join(text)


def read_data_from_txt(data_dir, split):
    """Read text and label from the data directory."""
    txt_file = os.path.join(data_dir, f'{split}_raw_texts.txt')
    label_file = os.path.join(data_dir, f'{split}_labels.txt')
    with open(txt_file, 'r') as f:
        text = f.readlines()
    with open(label_file, 'r') as f:
        label = f.readlines()

    return pd.DataFrame(data={
        'text': text,
        'label': label
    })


def preprocess(text, vocab=None):
    """Preprocess text data from AttentionXML dataset."""
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    text = re.sub('\\s+', ' ', text)
    text = text.lower().strip()
    if vocab is not None:
        text = [t for t in text.split() if t in vocab]
    return ' '.join(text)


def process_txt_data(data_dir, output_dir, vocab_file=None):
    """Preprocess data from AttentionXML such as `AmazonCat-13K`, `Wiki10-31K`, and `EUR-Lex`.
    """
    vocab = None
    if vocab_file is not None:
        with open(vocab_file, 'r') as f:
            vocab = set([v.strip() for v in f.readlines()])
        print(f'Load {len(vocab)} vocab from {vocab_file}.')

    # build train, test and label dict
    for split in ['train', 'test']:
        df = read_data_from_txt(data_dir, split)
        df['text'] = df['text'].apply(lambda x: preprocess(x, vocab))
        df['label'] = df['label'].str.strip()

        start_time = time.time()
        out_path = os.path.join(output_dir, f'{split}.txt')
        print(f'Writing {len(df)} {split} data to {out_path}.')
        df.to_csv(out_path, sep='\t', header=False, columns=['label', 'text'])

        print(f'Finish writing {len(df)} {split} data to {out_path}.')
        print(f'Time: {time.time() - start_time:.2f} sec')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir', default='/home/eleven/xml-cnn-study-code/data/EUR-Lex',
        help='Path to data directory (default: %(default)s).')
    parser.add_argument(
        '--vocab_file', default='/home/eleven/XML-CNN/data/EUR-Lex.bow/Xf.txt',
        help='Path to vocabulary file (i.e. Xf.txt) (default: %(default)s).')
    parser.add_argument(
        '--output_dir', help='Path to directory with train_bow.txt, test_bow.txt (default: %(default)s).')
    parser.add_argument(
        '--gen_train_test', action='store_true', help='Generate train.txt and text.txt from *_labels.txt and *_raw_texts.txt.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.gen_train_test:
        process_txt_data(args.data_dir, args.output_dir, args.vocab_file)
    else:
        # Read Xf.txt
        with open(args.vocab_file, 'r') as f:
            vocabs = [v.strip() for v in f.readlines()]
        vocabs = set(vocabs)
        print(f'Read {len(vocabs)} from {args.vocab_file}.')

        # Output train_bow.txt, test_bow.txt
        for split in ['train', 'test']:
            path = os.path.join(args.data_dir, f'{split}.txt')
            df = pd.read_csv(path, sep='\t', header=None, error_bad_lines=False, warn_bad_lines=True)
            df[2] = df[2].apply(lambda text: remove_vocab_xf(text, vocabs))
            output_path = os.path.join(args.output_dir, f'{split}_bow.txt')
            df.to_csv(output_path, sep='\t', index=False, header=False)
            print(f'Output {len(df)} {split} data to {output_path}.')


main()
