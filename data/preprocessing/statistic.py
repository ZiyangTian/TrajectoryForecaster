import os
import numpy as np
import pandas as pd


def count_file_lens(files, to_log=None):
    counts = []
    for file in files:
        with open(file, 'r') as f:
            count = -1
            for count, _ in enumerate(f):
                pass
            count += 1
            counts.append(count)
    if to_log is not None:
        with open(to_log, 'w') as f:
            for count in counts:
                f.write(str(count) + '\n')
    return counts


def wash_file_lens(files, wash_less_than):
    counts = []
    for file in files:
        with open(file, 'r') as f:
            count = -1
            for count, _ in enumerate(f):
                pass
            count += 1
            counts.append(count)
    for file, count in zip(files, counts):
        if count < wash_less_than:
            os.remove(file)
            print(file)


def cut_csv_files(csv_files, length, cut_from='tail', multiple_mode=False, **kwargs):
    invalids = []
    for file in csv_files:
        df = pd.read_csv(file, **kwargs)
        len_df = len(df)
        if len_df < length:
            invalids.append(file)
            os.remove(file)
        else:
            target_length = len_df - len_df % length if multiple_mode else length
            if len_df > target_length:
                if cut_from == 'tail':
                    df[:target_length].to_csv(header=False, index=False)
                elif cut_from == 'head':
                    df[-target_length:].to_csv(header=False, index=False)
                elif cut_from == 'random':
                    start_index = np.random.uniform(0, len_df - target_length + 1)
                    df[start_index: start_index + target_length].to_csv(header=False, index=False)
                else:
                    raise ValueError('Invalid `cut_from` value: {}.'.format(cut_from))
    return invalids


def main():
    import tensorflow as tf
    files = tf.io.gfile.glob('/Users/Tianziyang/projects/AnomalyDetection/data/test/raw/*.csv')
    # counts = count_file_lens(files, to_log='/Users/Tianziyang/projects/AnomalyDetection/data/train/filelen.csv')
    wash_file_lens(files, 5)


if __name__ == '__main__':
    main()
