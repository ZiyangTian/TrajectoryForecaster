import numpy as np
import pandas as pd
import tensorflow as tf

from data.anomaly_detection import generator


TRAIN_PATTERN = '/Users/Tianziyang/projects/AnomalyDetection/data/train/raw/*.csv'
TEST_PATTERN = '/Users/Tianziyang/projects/AnomalyDetection/data/test/raw/*.csv'
COLUMNS = ['t', 'x', 'y', 'z',
           'distance_anomaly', 'height_anomaly', 'high_speed_anomaly', 'low_speed_anomaly', 'airline_anomaly']
SEQUENCE_LEN = 5


def parse_data(pattern):
    files = tf.io.gfile.glob(pattern)
    features, labels = [], []

    for file in files:
        df = pd.read_csv(file)
        if list(df.columns) != COLUMNS:
            raise ValueError('.')
        for i in range(0, 4):
            df[COLUMNS[i]] = df[COLUMNS[i]].astype(np.float)
        for i in range(4, 9):
            df[COLUMNS[i]] = df[COLUMNS[i]].astype(np.int)
        inputs_df = np.array(df[COLUMNS[0: 4]], dtype=np.float)
        outputs_df = np.array(df[COLUMNS[4:]], dtype=np.int)
        len_df = len(df)
        num_examples = len_df - SEQUENCE_LEN + 1

        features.append(np.stack([inputs_df[i: i + num_examples] for i in range(seq_len)], axis=1))
        labels.append(outputs_df[4:])

    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def generate_data():
    train_features, train_labels = parse_data(TRAIN_PATTERN)
    test_features, test_labels = parse_data(TEST_PATTERN)
    gen = generator.TrajectoryGenerator(
        dt=10,
        num_normal_safe_airlines=8,
        num_normal_alert_airlines=2,
        prob_normal_safe=0.3,
        prob_normal_alert=0.1,
        prob_anomaly_straight=0.2,
        prob_anomaly_circle=0.2,
        prob_anomaly_random=0.2,
        prob_distance_anomaly=0.3,
        prob_height_anomaly=0.3,
        prob_high_speed_anomaly=0.6,
        prob_low_speed_anomaly=0.1,
        prob_airline_anomaly=0.6)
    trajectories = gen.generate(3000, save_in='/Users/Tianziyang/projects/AnomalyDetection/data/train', verbose=True)
    trajectories = gen.generate(300, save_in='/Users/Tianziyang/projects/AnomalyDetection/data/test', verbose=True)

    # generator.display(gen, trajectories)
    # circle1 = generator.generate_circular_trajectory(160, np.pi / 6, 200, 7, 0.25)
    # circle2 = generator.generate_circular_trajectory(170, np.pi, 210, 7, 0.2)
    # circle3 = generator.generate_circular_trajectory(180, np.pi / 2, 205, 7, 0.3)
    # other_test = generator.generate_random_trajectory(0.25, 0.8, 0, 0.01, 0.01)
    # other_test.to_csv('other_test.csv', index=False)


def test_example():
    file = '/Users/Tianziyang/projects/AnomalyDetection/data/train/raw/12.csv'
    features, labels = parse_data(file,  5)


if __name__ == '__main__':
    main()
