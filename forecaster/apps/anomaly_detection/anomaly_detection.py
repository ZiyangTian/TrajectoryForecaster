import os
import tensorflow as tf

from forecaster.apps import base
from forecaster.data import sequence
from forecaster.models import networks
from forecaster.models import optimizers


def make_dataset(pattern,
                 raw_data_spec: sequence.RawDataSpec,
                 feature_column_names,
                 label_column_names,
                 sequence_length,
                 shift,
                 stride,
                 batch_size,
                 repeat_infinitely=False,
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
        dataset = dataset.map(
            lambda feature_dict: (feature_dict['features'], feature_dict['labels']),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if repeat_infinitely:
            dataset = dataset.repeat()
        if shuffle_buffer_size is not None:
            dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.batch(batch_size)
    return dataset


class AnomalyDetector(tf.keras.Model):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_attention_heads,
                 conv_kernel_size,
                 num_anomaly_types,
                 numeric_normalizer_fn=None,
                 input_shape=None):
        super(AnomalyDetector, self).__init__()
        self._encoder = networks.SequenceEncoder(
            num_layers=num_layers, d_model=d_model,
            num_attention_heads=num_attention_heads, conv_kernel_size=conv_kernel_size,
            numeric_normalizer_fn=numeric_normalizer_fn, numeric_restorer_fn=None, name=None)
        self._head_dense1 = tf.keras.layers.Dense(1)
        self._head_dense2 = tf.keras.layers.Dense(num_anomaly_types, activation='sigmoid')
        self._input_shape = input_shape

    def call(self, inputs, **kwargs):
        inputs.set_shape([None, self._input_shape[0], self._input_shape[1]])
        encoded = self._encoder(inputs)
        with tf.name_scope('head'):
            dense1 = tf.squeeze(self._head_dense1(tf.transpose(encoded, (0, 2, 1))), axis=-1)
            outputs = self._head_dense2(dense1)
        return outputs


@base.app('anomaly_detection')
class AnomalyDetection(base.App):
    def build(self):
        data_config = self._config.data
        model_config = self._config.model
        normalizer_mean = list(map(
            lambda k: self._config.raw_data.features.__getattr__(k).mean, self._config.data.features))
        normalizer_std = list(map(
            lambda k: self._config.raw_data.features.__getattr__(k).std, self._config.data.features))

        def numeric_normalizer_fn(features):
            with tf.name_scope('numeric_normalizer_fn'):
                mean = tf.constant(normalizer_mean, dtype=tf.float32)
                std = tf.constant(normalizer_std, dtype=tf.float32)
                ans = (features - mean) / std
            return ans

        model = AnomalyDetector(
            model_config.num_layers, model_config.d_model,
            model_config.num_attention_heads,
            model_config.conv_kernel_size,
            len(data_config.labels),
            numeric_normalizer_fn=numeric_normalizer_fn,
            input_shape=(data_config.sequence_length, len(data_config.features)))
        model.compile(
            optimizers.get_optimizer(model_config.optimizer),
            loss='categorical_crossentropy',
            metrics=['binary_accuracy'])
        return model

    @property
    def train_dataset(self):
        data_config = self._config.data
        train_config = self._config.run.train
        return make_dataset(
            os.path.join(self._config.raw_data.train_dir, '*.csv'),
            sequence.RawDataSpec.from_config(self._config.raw_data),
            data_config.features,
            data_config.labels,
            data_config.sequence_length,
            train_config.data_shift,
            data_config.stride,
            train_config.batch_size,
            repeat_infinitely=True,
            shuffle_buffer_size=train_config.shuffle_buffer_size,
            name='train_dataset')

    @property
    def valid_dataset(self):
        data_config = self._config.data
        eval_config = self._config.run.eval
        return make_dataset(
            os.path.join(self._config.raw_data.eval_dir, '*.csv'),
            sequence.RawDataSpec.from_config(self._config.raw_data),
            data_config.features,
            data_config.labels,
            data_config.sequence_length,
            eval_config.data_shift,
            data_config.stride,
            eval_config.batch_size,
            repeat_infinitely=False,
            shuffle_buffer_size=None,
            name='valid_dataset')

    @property
    def test_dataset(self):
        data_config = self._config.data
        predict_config = self._config.run.predict
        return make_dataset(
            os.path.join(self._config.raw_data.test_dir, '*.csv'),
            sequence.RawDataSpec.from_config(self._config.raw_data),
            data_config.features,
            data_config.labels,
            data_config.sequence_length,
            predict_config.data_shift,
            data_config.stride,
            predict_config.batch_size,
            repeat_infinitely=False,
            shuffle_buffer_size=None,
            name='test_dataset')
