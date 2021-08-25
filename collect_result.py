import argparse
import glob
import json
import os

import pandas as pd
from matplotlib import pyplot as plt

# INDEX = ['model_name', 'num_filter_per_size', 'learning_rate', 'dropout', 'dropout2', 'pool_size']
# INDEX = ['model_name', 'num_filter_per_size', 'learning_rate', 'dropout', 'dropout2', 'pool_size', 'init_weight']
# INDEX = ['model_name', 'num_filter_per_size', 'pool_size', 'init_weight']
# INDEX = ['model_name', 'batch_size', 'max_seq_length', 'num_pool', 'shuffle']
INDEX = ['model_name', 'learning_rate', 'num_filter_per_size', 'filter_sizes', 'dropout']
# COL = ['Micro-F1', 'Macro-F1', 'Macro*-F1', 'P@1', 'P@3', 'P@5']
COL = ['Micro-F1', 'Macro-F1', 'P@1', 'P@3', 'P@5']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', default='./runs')
    parser.add_argument('--data_name', default='EUR-Lex')
    parser.add_argument('--output', default='./result.xlsx')
    return parser.parse_args()


def box_plots(orig_df):
    orig_df =  orig_df.groupby(level=orig_df.index.names).mean()
    for index in INDEX:
        df = orig_df.reset_index(index).reset_index(drop=True)
        boxplot = df.boxplot(by=index)
        plt.savefig(f'imgs/{index}.png')
        plt.clf()


def main():
    args = parse_args()
    records = []
    for log_path in glob.glob(os.path.join(args.logdir, f'{args.data_name}_*', 'logs.json')):
        with open(log_path) as fp:
            data = json.load(fp)
        if 'test' not in data:
            continue
        data['config']['filter_sizes'] = '_'.join(map(str, data['config']['filter_sizes']))
        records.append({**data['config'], **data['test'][-1]})
    print(f'total {len(records)}')

    df = pd.DataFrame.from_records(records)
    df = df[INDEX + COL]
    df = df.set_index(INDEX)

    # box_plots(df)

    count = df.groupby(level=df.index.names).agg(['count']).iloc[:,0]
    avg_df = df.groupby(level=df.index.names).agg(['mean', 'std'])
    df = pd.concat([count, avg_df], axis=1)
    # df = df.sort_values(('Macro-F1', 'mean'), ascending=False)
    df = df.sort_values(('P@1', 'mean'), ascending=False)

    import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
    df = df.reset_index(col_level=1)
    with pd.ExcelWriter(args.output, mode='w') as writer:
        df.to_excel(writer, sheet_name=args.data_name)


if __name__ == '__main__':
    main()
