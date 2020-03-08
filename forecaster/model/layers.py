""" `tf.keras.layers.Layer` subclasses. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from forecaster.math.diff import diff_1_pad


class Uniform(tf.keras.layers.Layer):
    def __init__(self, centre=None, bound=None,
                 name=None, dtype=tf.float32, **kwargs):
        super(Uniform, self).__init__(name=name, dtype=dtype, **kwargs)
        self._centre = centre or 1.
        self._bound = bound or 1.

    def call(self, inputs, **kwargs):
        centre, bound = tf.convert_to_tensor(self._centre), tf.convert_to_tensor(self._bound)
        return (inputs - centre) / bound


class Restore(tf.keras.layers.Layer):
    def __init__(self, centre=None, bound=None,
                 name=None, dtype=tf.float32, **kwargs):
        super(Restore, self).__init__(name=name, dtype=dtype, **kwargs)
        self._centre = centre or 1.
        self._bound = bound or 1.

    def call(self, inputs, **kwargs):
        centre, bound = tf.convert_to_tensor(self._centre), tf.convert_to_tensor(self._bound)
        return inputs * bound + centre


class UniformDiff(tf.keras.layers.Layer):
    def __init__(self, centre=None, bound=None, diff_axis=-1,
                 name=None, dtype=tf.float32, **kwargs):
        super(UniformDiff, self).__init__(name=name, dtype=dtype, **kwargs)
        self._diff_axis = diff_axis
        self._uniform = Uniform(centre, bound, dtype=dtype, **kwargs)
        self._concatenate = tf.keras.layers.Concatenate(axis=-1)

    def call(self, inputs, **kwargs):
        uniformed_inputs = self._uniform(inputs)
        diff_inputs = diff_1_pad(uniformed_inputs, axis=self._diff_axis)
        return self._concatenate([uniformed_inputs, diff_inputs])


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, attention_dim, output_seq_len,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(SelfAttention, self).__init__(
            trainable=trainable, name=name or 'self_attention', dtype=dtype, **kwargs)
        self._wh = tf.keras.layers.Dense(output_seq_len)
        self._wq = tf.keras.layers.Dense(attention_dim)
        self._reshape = tf.keras.layers.Reshape([output_seq_len, attention_dim])
        self._wk = tf.keras.layers.Dense(attention_dim)
        self._wv = tf.keras.layers.Dense(attention_dim)
        self._attention_dim = attention_dim

    def call(self, inputs, **kwargs):
        hidden_outputs = inputs[0]
        hidden_state = tf.stack(inputs[1:], axis=1)
        hidden_state = tf.transpose(hidden_state, perm=(0, 2, 1))
        q = self._wh(hidden_state)
        q = tf.transpose(q, perm=(0, 2, 1))
        q = self._wq(q)
        q = self._reshape(q)
        k = self._wk(hidden_outputs)
        v = self._wv(hidden_outputs)
        sh = tf.matmul(q, k, transpose_b=True)
        attention_dim_float = tf.cast(self._attention_dim, dtype=tf.float32)
        attention_weights = tf.nn.softmax(sh / tf.sqrt(attention_dim_float), axis=-1)
        outputs = tf.matmul(attention_weights, v)
        return outputs


class DotMatch(tf.keras.layers.Layer):
    def __init__(self, m, n):
        super(DotMatch, self).__init__()
        self._w = self.add_weight(
            'w', (m, n), dtype=tf.float32, trainable=True, initializer=tf.random_normal_initializer)

    def call(self, inputs, **kwargs):
        fore_state, post_state = inputs
        return tf.einsum('jm,mn,kn->kj', fore_state, self._w, post_state)
