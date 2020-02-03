"""  """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import collections
import tensorflow as tf

from forecaster.model import layers


class ModelSpec(collections.namedtuple(
    'ModelSpec', (
        'input_seq_len',
        'output_seq_len',
        'output_dim',
        'nn_params'))):
    def __new__(cls,
                input_seq_len,
                output_seq_len,
                output_dim,
                nn_params,
                *args,
                **kwargs):
        return super(ModelSpec, cls).__new__(
            cls,
            input_seq_len,
            output_seq_len,
            output_dim,
            copy.deep_copy(nn_params))

    @property
    def uniform_centre(self):
        return self.nn_params.get('uniform_centre', 0.)

    @property
    def uniform_bound(self):
        return self.nn_params.get('uniform_bound', 1.)


class LSTMModel(ModelSpec, tf.keras.Model):
    def __init__(self, input_seq_len, output_seq_len, output_dim, nn_params):
        super(LSTMModel, self).__init__(input_seq_len, output_seq_len, output_dim, nn_params)
        hidden_size = self.nn_params['hidden_size']
        hidden_layers = self.nn_params['hidden_layers']

        self._uniform_diff = layers.UniformDiff(
            self.uniform_centre, self.uniform_bound, diff_axis=1, name='uniform_diff')
        self._rnn = tf.keras.layers.RNN(
            [tf.keras.layers.LSTMCell(hidden_size) for _ in range(hidden_layers)],
            return_sequences=True, name='deep_lstm')
        self._dense = tf.keras.layers.Dense(self.output_dim, name='dense')
        self._restore = layers.Restore(
            self.uniform_centre[:self.output_dim], self.uniform_bound[:self.output_dim], name='restore')

    def call(self, inputs, training=None, mask=None):
        outputs = self._uniform_diff(inputs)
        outputs = self._rnn(outputs)
        outputs = self._dense(outputs[..., -self.output_seq_len:, :])
        predictions = self._restore(outputs)
        return outputs, predictions


class SelfAttentionModel(ModelSpec, tf.keras.Model):
    def __init__(self, input_seq_len, output_seq_len, output_dim, nn_params):
        super(SelfAttentionModel, self).__init__(input_seq_len, output_seq_len, output_dim, nn_params)
        hidden_size = self.nn_params['hidden_size']

        self._uniform_diff = layers.UniformDiff(
            self.uniform_centre, self.uniform_bound, diff_axis=1, name='uniform_diff')
        self._rnn = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True), name='rnn')
        self._attention = layers.SelfAttention(hidden_size, self.output_seq_len, name='self_attention')
        self._dense = tf.keras.layers.Dense(self.output_dim, name='dense')
        self._restore = layers.Restore(
            self.uniform_centre[:self.output_dim], self.uniform_bound[:self.output_dim], name='restore')

    def call(self, inputs, training=None, mask=None):
        outputs = self._uniform_diff(inputs)
        outputs = self._rnn(outputs)
        outputs = self._attention(outputs)
        outputs = self._dense(outputs)
        predictions = self._restore(outputs)
        return outputs, predictions

