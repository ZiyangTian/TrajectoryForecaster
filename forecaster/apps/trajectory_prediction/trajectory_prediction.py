import os
import tensorflow as tf

from forecaster.apps import base
from forecaster.data import sequence
from forecaster.models import layers
from forecaster.models import metrics
from forecaster.models import networks
from forecaster.models import optimizers


def make_dataset(pattern,
                 raw_data_spec: sequence.RawDataSpec,
                 trajectory_feature_names,
                 other_feature_names,
                 input_sequence_length,
                 output_sequence_length,
                 shift,
                 stride,
                 batch_size,
                 repeat_infinitely=False,
                 shuffle_buffer_size=None,
                 name=None):
    data_files = tf.io.gfile.glob(pattern)
    full_column_spec = sequence.SeqColumnsSpec(
        list(trajectory_feature_names) + list(other_feature_names),
        input_sequence_length + output_sequence_length,
        group=True, new_names='full_sequence')

    def generate_targes(feature_dict):
        inputs = feature_dict['full_sequence']
        targets = inputs[:, -output_sequence_length:, :len(trajectory_feature_names)]
        return inputs, targets

    with tf.name_scope(name or 'make_dataset'):
        dataset = sequence.sequence_dataset(
            [full_column_spec],
            data_files, raw_data_spec,
            shift=shift, stride=stride, shuffle_files=True,
            name='sequence_dataset')
        dataset = dataset.map(generate_targes, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if repeat_infinitely:
            dataset = dataset.repeat()
        if shuffle_buffer_size is not None:
            dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.batch(batch_size)

    return dataset


class TrajectoryPredictor(tf.keras.Model):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_attention_heads,
                 conv_kernel_size,
                 output_sequence_length,
                 num_outputs,
                 mask,
                 numeric_normalizer_fn=None,
                 numeric_restorer_fn=None,
                 input_shape=None):
        super(TrajectoryPredictor, self).__init__()
        self._mask = mask
        self._encoder = networks.SequenceEncoder(
            num_layers=num_layers, d_model=d_model,
            num_attention_heads=num_attention_heads, conv_kernel_size=conv_kernel_size,
            numeric_normalizer_fn=numeric_normalizer_fn, name=None)
        self._head_dense = tf.keras.layers.Dense(num_outputs)
        self._input_shape = input_shape
        self._output_sequence_length = output_sequence_length
        self._numeric_restorer = layers.FunctionWrapper(
            tf.identity if numeric_restorer_fn is None else numeric_restorer_fn,
            name='numeric_restorer')

    def call(self, inputs, **kwargs):
        encoded = self._encoder(inputs, mask=self._mask)
        with tf.name_scope('head'):
            dense1 = tf.squeeze(self._head_dense1(tf.transpose(encoded, (0, 2, 1))), axis=-1)
            outputs = self._head_dense2(dense1)[:, -self._output_sequence_length:, :]
            restored_outputs = self._numeric_restorer(outputs)
        return outputs, restored_outputs


@base.app('trajectory_prediction')
class TrajectoryPrediction(base.App):
    def build(self):
        data_config = self._config.data
        model_config = self._config.model
        normalizer_mean = list(map(
            lambda k: self._config.raw_data.features.__getattr__(k).mean, self._config.data.features))
        normalizer_std = list(map(
            lambda k: self._config.raw_data.features.__getattr__(k).std, self._config.data.features))
        num_trajectory_features = len(data_config.trajectory_features)
        num_other_features = len(data_config.other_features)
        num_features = num_trajectory_features + num_other_features

        input_mask = tf.concat([
            tf.ones((data_config.input_sequence_length, num_features), dtype=tf.int32),
            tf.zeros((data_config.output_sequence_length, num_features), dtype=tf.int32)],
            axis=0)

        def numeric_normalizer_fn(features):
            with tf.name_scope('numeric_normalizer_fn'):
                mean = tf.convert_to_tensor(normalizer_mean, dtype=tf.float32)
                std = tf.convert_to_tensor(normalizer_std, dtype=tf.float32)
                ans = (features - mean) / std
            return ans

        def numeric_restorer_fn(features):
            with tf.name_scope('numeric_restorer_fn'):
                mean = tf.convert_to_tensor(normalizer_mean, dtype=tf.float32)
                std = tf.convert_to_tensor(normalizer_std, dtype=tf.float32)
                ans = features * std + mean
            return ans

        model = TrajectoryPredictor(
            model_config.num_layers, model_config.d_model,
            model_config.num_attention_heads,
            model_config.conv_kernel_size,
            data_config.output_sequence_length,
            len(data_config.labels),
            input_mask,
            numeric_normalizer_fn=numeric_normalizer_fn,
            numeric_restorer_fn=numeric_restorer_fn,
            input_shape=(data_config.sequence_length, len(data_config.features)))

        model.compile(
            optimizers.get_optimizer(model_config.optimizer),
            loss='mse',
            metrics=[
                metrics.DestinationDeviation(),
                metrics.MinDeviation(),
                metrics.MeanDeviation(),
                metrics.MaxDeviation()],
            loss_weights=(1, 0))
        return model

    @property
    def train_dataset(self):
        data_config = self._config.data
        train_config = self._config.run.train
        return make_dataset(
            os.path.join(self._config.raw_data.train_dir, '*.csv'),
            sequence.RawDataSpec.from_config(self._config.raw_data),
            data_config.trajectory_features,
            data_config.other_features,
            data_config.input_sequence_length,
            data_config.output_sequence_length,
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
            data_config.trajectory_features,
            data_config.other_features,
            data_config.input_sequence_length,
            data_config.output_sequence_length,
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
            data_config.trajectory_features,
            data_config.other_features,
            data_config.input_sequence_length,
            data_config.output_sequence_length,
            predict_config.data_shift,
            data_config.stride,
            predict_config.batch_size,
            repeat_infinitely=False,
            shuffle_buffer_size=None,
            name='test_dataset')
