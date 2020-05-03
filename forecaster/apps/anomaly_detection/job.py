import os
import numpy as np
import pandas as pd
import tensorflow as tf

from forecaster.data import sequence
from forecaster.models import networks
from forecaster.run import monitor

TRAIN_PATTERN = '/Users/Tianziyang/projects/AnomalyDetection/data/train/raw/*.csv'
TEST_PATTERN = '/Users/Tianziyang/projects/AnomalyDetection/data/test/raw/*.csv'
COLUMNS = ['t', 'x', 'y', 'z',
           'distance_anomaly', 'height_anomaly', 'high_speed_anomaly', 'low_speed_anomaly', 'airline_anomaly']
SEQUENCE_LEN = 5


def make_dataset(pattern,
                 raw_data_spec: sequence.RawDataSpec,
                 feature_column_names,
                 label_column_names,
                 sequence_length,
                 shift,
                 stride,
                 batch_size,
                 num_epochs,
                 shuffle_buffer_size=None,
                 name=None):
    data_files = tf.io.gfile.glob(pattern)
    feature_column_spec = sequence.SeqColumnsSpec(
        feature_column_names, sequence_length, group=True, new_names='features')
    label_column_spec = sequence.ReducingColumnsSpec(
        label_column_names, rsv_pos=sequence_length - 1, group=True, new_names='labels')
    with tf.name_scope(name or 'make_dataset'):
        dataset = sequence.sequence_dataset(
            [feature_column_spec, label_column_spec],
            data_files, raw_data_spec,
            shift=shift, stride=stride, shuffle_files=True,
            name='sequence_dataset')
        dataset = dataset.repeat(num_epochs)
        if shuffle_buffer_size is not None:
            dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.batch(batch_size)
    return dataset


class AnomalyDetection(monitor.Job):
    def build(self, warm_start_from=None):
        pass

    @property
    def train_input_fn(self):
        data_config = self._config.data
        train_config = self._config.run.train

        def input_fn():
            return make_dataset(
                os.path.join(self._config.raw_data.train_dir, '*.csv'),
                self._raw_data_spec,
                data_config.features,
                data_config.labels,
                data_config.sequence_length,
                train_config.data_shift,
                data_config.stride,
                train_config.batch_size,
                train_config.num_epochs,
                shuffle_buffer_size=train_config.shuffle_buffer_size,
                name='train_input_fn')

        return input_fn

    @property
    def eval_input_fn(self):
        data_config = self._config.data
        eval_config = self._config.run.eval

        def input_fn():
            return make_dataset(
                os.path.join(self._config.raw_data.eval_dir, '*.csv'),
                self._raw_data_spec,
                data_config.features,
                data_config.labels,
                data_config.sequence_length,
                eval_config.data_shift,
                data_config.stride,
                eval_config.batch_size,
                1,
                shuffle_buffer_size=None,
                name='eval_input_fn')

        return input_fn

    @property
    def predict_input_fn(self):
        data_config = self._config.data
        predict_config = self._config.run.predict

        def input_fn():
            return make_dataset(
                os.path.join(self._config.raw_data.test_dir, '*.csv'),
                self._raw_data_spec,
                data_config.features,
                data_config.labels,
                data_config.sequence_length,
                predict_config.data_shift,
                data_config.stride,
                predict_config.batch_size,
                1,
                shuffle_buffer_size=None,
                name='predict_input_fn')

        return input_fn


class Detector(tf.keras.Model):
    def __init__(self):
        super(Detector, self).__init__()
        self._encoder = networks.SequenceEncoder(
            num_layers=2, d_model=32, num_attention_heads=4, conv_kernel_size=3,
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
