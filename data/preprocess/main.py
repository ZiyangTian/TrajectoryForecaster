"""Reserve."""
import os
import glob

from data.preprocess import statistic


def main():
    raw_data_dir = '/Users/Tianziyang/projects/AnomalyDetection/data/train/raw'

    n = 1

    if n == 0:
        statistic.count_file_lens(
            glob.glob(os.path.join(raw_data_dir, '*.csv')),
            to_log='/Users/Tianziyang/Desktop/log.csv')
    if n == 1:
        statistic.cut_csv_files(
            glob.glob(os.path.join(raw_data_dir, '*.csv')), 5, cut_from='random', multiple_mode=True)


if __name__ == '__main__':
    main()
