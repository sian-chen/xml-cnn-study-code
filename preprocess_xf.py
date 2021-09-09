import argparse
import os
import pandas as pd


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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

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
