import numpy as np
import pandas as pd
import tensorflow as tf

from forecaster.models import networks

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

        features.append(np.stack([inputs_df[i: i + num_examples] for i in range(SEQUENCE_LEN)], axis=1))
        labels.append(outputs_df[4:])

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    return features[:, :, 1:], labels


class Detector(tf.keras.Model):
    def __init__(self):
        super(Detector, self).__init__()
        self._encoder = networks.SequenceEncoder(
            num_layers=2, d_model=16, num_attention_heads=4, conv_kernel_size=3,
            numeric_normalizer_fn=lambda x: x / [300., 300., 10.], numeric_restorer_fn=None, name=None)
        self._head_dense1 = tf.keras.layers.Dense(1)
        self._head_dense2 = tf.keras.layers.Dense(5, activation='sigmoid')

    def call(self, inputs, **kwargs):
        encoded = self._encoder(inputs)
        dense1 = tf.squeeze(self._head_dense1(tf.transpose(encoded, (0, 2, 1))), axis=-1)
        outputs = self._head_dense2(dense1)
        return outputs


def main():
    train_features, train_labels = parse_data(TRAIN_PATTERN)
    test_features, test_labels = parse_data(TEST_PATTERN)
    model = Detector()
    model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath='/Users/Tianziyang/projects/AnomalyDetection/data/saved/ckpts'),
        tf.keras.callbacks.TensorBoard(log_dir='/Users/Tianziyang/projects/AnomalyDetection/data/saved/tensorboard',
                                       update_freq='batch')
    ]
    model.fit(
        train_features, train_labels,
        batch_size=32, epochs=100, shuffle=True,
        validation_data=(test_features, test_labels),
        callbacks=callbacks)


if __name__ == '__main__':
    main()
