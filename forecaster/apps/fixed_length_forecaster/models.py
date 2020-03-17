"""  """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import collections
import tensorflow as tf

from forecaster.apps import base as apps_base
from forecaster.model import layers
from forecaster.model import optimizers
from forecaster.model import metrics


class _UniformedMSE(tf.keras.losses.Loss):
    def __init__(self, uniform_centre=None, uniform_bound=None, name=None):
        super(_UniformedMSE, self).__init__(name=name)
        self._uniform_centre = uniform_centre
        self._uniform_bound = uniform_bound

    def call(self, y_true, y_pred):
        y_true = layers.Uniform(centre=self._uniform_centre, bound=self._uniform_bound)(y_true)
        return tf.keras.losses.mse(y_true, y_pred)


@apps_base.app_register('FixedLengthForecaster')
class FixedLengthForecaster(apps_base.AbstractSequencesApp):
    def build(self):
        model_config = self._config.model
        uniform_centre = ...
        uniform_bound = ...
        restore_centre = ...
        restore_bound = ...
        num_features = ...
        output_dim = ...

        inputs = tf.keras.Input((model_config.input_seq_len, num_features), name='input_layer')
        outputs = tf.keras.Sequential([
            layers.UniformDiff(
                uniform_centre, uniform_bound, diff_axis=1, name='uniform_diff'),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(model_config.hidden_size, return_sequences=True, return_state=True), name='rnn'),
            layers.SelfAttention(
                model_config.hidden_size, model_config.output_seq_len, name='self_attention'),
            tf.keras.layers.Dense(
                output_dim, name='dense')
        ])(inputs)
        predictions = layers.Restore(restore_centre, restore_bound, name='restore')(outputs)
        self._model = tf.keras.Model(inputs, (outputs, predictions))

        self._model.compile(
            optimizer=optimizers.get_optimizer(dict(model_config.optimizer)),
            loss=[_UniformedMSE(uniform_centre, uniform_bound), None],
            metrics=[
                None,
                [metrics.get_metrics(
                    'max_deviation',
                    'mean_deviation',
                    'min_deviation',
                    'destination_deviation')]])

    @property
    def train_dataset(self):
        return None

    @property
    def eval_dataset(self):
        return None

    @property
    def predict_dataset(self):
        return None
