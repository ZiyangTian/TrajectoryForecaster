"""Fixed length forecaster models and app."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import tensorflow as tf

from forecaster.apps import base as apps_base
from forecaster.data import sequence
from forecaster.models import layers
from forecaster.models import optimizers
from forecaster.models import metrics


class _UniformedMSE(tf.keras.losses.Loss):
    def __init__(self, uniform_centre=None, uniform_bound=None, name=None):
        super(_UniformedMSE, self).__init__(name=name)
        self._uniform_centre = uniform_centre
        self._uniform_bound = uniform_bound

    def call(self, y_true, y_pred):
        y_true = layers.Uniform(centre=self._uniform_centre, bound=self._uniform_bound)(y_true)
        return tf.keras.losses.mse(y_true, y_pred)


@apps_base.model('FixedLengthForecasterModel', short_name='fixed_len')
class FixedLengthForecasterModel(tf.keras.Model):
    def __init__(self,
                 output_seq_len,
                 output_dim,
                 hidden_size,
                 uniform_centre=None,
                 uniform_bound=None,
                 restore_centre=None,
                 restore_bound=None,
                 dropout=None,
                 name=None):
        super(FixedLengthForecasterModel, self).__init__(name=name or 'FixedLengthForecasterModel')

        self._uniform_diff = layers.UniformDiff(uniform_centre, uniform_bound, diff_axis=1, name='uniform_diff')
        self._rnn = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True), name='rnn')
        self._attention = layers.SelfAttention(hidden_size, output_seq_len, name='self_attention')
        self._layer_norm = tf.keras.layers.LayerNormalization()
        self._dropout = tf.keras.layers.Dropout(1. if dropout is None else dropout)
        self._dense = tf.keras.layers.Dense(output_dim, name='dense')
        self._restore = layers.Restore(restore_centre, restore_bound, name='restore')

    def call(self, inputs, training=None, mask=None):
        del mask
        current_layer = self._uniform_diff(inputs)
        current_layer = self._rnn(current_layer)
        current_layer = self._attention(current_layer)
        current_layer = self._layer_norm(current_layer)
        current_layer = self._dropout(current_layer, training=training)
        outputs = self._dense(current_layer)
        predictions = self._restore(outputs)
        return outputs, predictions


@apps_base.app('FixedLengthForecaster', short_name='fixed_len')
class FixedLengthForecaster(apps_base.AbstractSequencesApp):

    def build(self):
        model_config = self._config.model
        output_dim, uniform_centre, uniform_bound, restore_centre, restore_bound = self._extract_uniform_data()

        self._model = FixedLengthForecasterModel(
            self._config.data.output_seq_len,
            output_dim,
            model_config.hidden_size,
            uniform_centre=uniform_centre,
            uniform_bound=uniform_bound,
            restore_centre=restore_centre,
            restore_bound=restore_bound,
            dropout=model_config.dropout)
        self._model.compile(
            optimizer=optimizers.get_optimizer(dict(model_config.optimizer)),
            loss=[_UniformedMSE(restore_centre, restore_bound), None],
            metrics=[
                [],
                [metrics.max_deviation, metrics.mean_deviation]])

    @property
    def train_dataset(self):
        train_config = self._config.run.train
        return self._make_dataset(
            tf.io.gfile.glob(os.path.join(self._config.raw_data.train_dir, '*')),
            train_config.batch_size,
            shuffle_buffer_size=train_config.shuffle_buffer_size,
            shuffle_files=True,
            shift=train_config.data_shift)

    @property
    def eval_dataset(self):
        eval_config = self._config.run.eval
        return self._make_dataset(
            tf.io.gfile.glob(os.path.join(self._config.raw_data.eval_dir, '*')),
            eval_config.batch_size,
            shuffle_buffer_size=eval_config.shuffle_buffer_size,
            shuffle_files=True,
            shift=eval_config.data_shift)

    @property
    def predict_dataset(self):
        return None

    def _extract_uniform_data(self):
        features_config = self._config.raw_data.features
        data_config = self._config.data

        input_features = data_config.input_features
        target_features = data_config.target_features
        output_dim = len(target_features)

        uniform_centre = list(map(lambda k: features_config[k]['mean'], input_features))
        uniform_bound = list(map(lambda k: features_config[k]['std'], input_features))
        restore_centre = list(map(lambda k: features_config[k]['mean'], target_features))
        restore_bound = list(map(lambda k: features_config[k]['std'], target_features))

        return output_dim, uniform_centre, uniform_bound, restore_centre, restore_bound

    def _make_dataset(self,
                      data_files,
                      batch_size,
                      shuffle_buffer_size=None,
                      shuffle_files=True,
                      shift=None):
        data_config = self._config.data

        input_columns_spec = sequence.SeqColumnsSpec(
            data_config.input_features,
            data_config.input_seq_len,
            new_names='input_sequence',
            group=True)
        target_columns_spec = sequence.SeqColumnsSpec(
            data_config.target_features,
            data_config.output_seq_len,
            new_names='target_sequence',
            group=True)
        dataset = sequence.sequence_dataset(
            [input_columns_spec, target_columns_spec],
            data_files, self._raw_data_spec, shuffle_files=shuffle_files, shift=shift)

        dataset = dataset.repeat()
        if shuffle_buffer_size is not None:
            dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.batch(batch_size).map(
            lambda feature_dict: (feature_dict['input_sequence'], feature_dict['target_sequence']))
        return dataset
