import os
import shutil


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


def main():
    import tensorflow as tf
    files = tf.io.gfile.glob('/Users/Tianziyang/projects/AnomalyDetection/data/test/raw/*.csv')
    # counts = count_file_lens(files, to_log='/Users/Tianziyang/projects/AnomalyDetection/data/train/filelen.csv')
    wash_file_lens(files, 5)


if __name__ == '__main__':
    main()
