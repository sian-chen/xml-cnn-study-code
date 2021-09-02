import os
import json
import glob

import pandas as pd

METRICS = ['P@1', 'P@3', 'P@5', 'Micro-Precision', 'Micro-Recall', 'Micro-F1',
           'Macro-F1', 'Another-Macro-F1']
METRICS = ['test_' + metric for metric in METRICS] + ['val_' + metric for metric in METRICS]
VAL_METRIC = {'EUR-Lex': 'P@1', 'Wiki10-31K': 'P@3', 'Amazon-670K': 'P@1', 'AmazonCat-13K': 'P@1'}


def read_exp_log(exp_log_path):
    with open(exp_log_path, 'r') as fp:
        log_data = json.load(fp)
    records = []
    for ckpt_s in log_data['checkpoints']:
        ckpt = json.loads(ckpt_s)
        record = {}
        for k, v in ckpt['config'].items():
            if isinstance(v, list):
                record[k] = '_'.join(map(str, v))
            elif isinstance(v, dict):
                pass
            else:
                record[k] = v
        model_name = os.path.basename(ckpt['config']['config'])[:-4]
        record['model_name'] = model_name
        if not ckpt['last_result']:
            continue
        for metric in METRICS:
            record[metric] = ckpt['last_result'][metric]
        records.append(record)
    df = pd.DataFrame.from_records(records)
    return df


def write_xlsx(df, path, sheet, mode):
    # nunique = df.nunique()
    # cols_to_drop = nunique[nunique <= 1].index
    # cols_to_drop = list(set(cols_to_drop) - set(['model_name', 'data_name']))
    # df = df.drop(cols_to_drop, axis=1)
    # cols = ['data_name', 'model_name'] + list(set(df.columns) - set(METRICS + ['model_name', 'data_name'])) + METRICS
    cols = ['model_name'] + METRICS
    df = df[cols]
    with pd.ExcelWriter(path, mode=mode) as writer:
        df.to_excel(writer, sheet_name=sheet, index=False)


def main():
    summaries = {}
    for exp_log_path in glob.glob('*/experiment_state-*.json'):
        print(exp_log_path)
        df = read_exp_log(exp_log_path)
        try:
            data_name = df['data_name'][0]
            model_name = df['model_name'][0]
        except Exception as e:
            continue

        val_metric = 'val_' + VAL_METRIC[data_name]
        df = df.sort_values(val_metric, ascending=False)
        # write_xlsx(df, 'result.xlsx', data_name + ' ' + model_name,
                   # 'w' if first else 'a')

        if data_name not in summaries:
            summaries[data_name] = []
        summaries[data_name].append(df.iloc[[0]])

    first = True
    for data_name, records in summaries.items():
        df = pd.concat(records, axis=0)
        test_metric = 'test_' + VAL_METRIC[data_name]
        df = df.sort_values(test_metric, ascending=False)
        write_xlsx(df, 'result.xlsx', data_name, 'w' if first else 'a')
        first = False


if __name__ == '__main__':
    main()
