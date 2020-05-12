import functools
import tensorflow as tf

from forecaster.models import layers
from forecaster.ops import diff


def scaled_dot_product_attention(q, k, v, name=None):
    """ Compute attention weights.
        Arguments
            q: query, shape == (..., seq_len_q, depth)
            k: key, shape == (..., sequence_length, depth)
            v: value, shape == (..., sequence_length, depth_v)
            mask: A float Tensor with shape broadcastable to
                  (..., seq_len_q, sequence_length). Defaults to None.
        Returns
            Outputs. Attention weights. (..., seq_len_q, depth_v)
        * q, k, v must have must have matching leading dimensions.
    """
    with tf.name_scope(name or 'scaled_dot_product_attention'):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)  # (..., seq_len_q, sequence_length)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_output = tf.matmul(attention_weights, v)
    return attention_output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name=None):
        super(MultiHeadAttention, self).__init__(name=name or 'multi_head_attention')
        self.num_heads = num_heads
        self.d_model = d_model
        if d_model % num_heads != 0:
            raise ValueError(
                '`d_model` must be divided by `num_heads`, got {} and {}'.format(d_model, num_heads))
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model, name='dense_q')
        self.wk = tf.keras.layers.Dense(d_model, name='dense_k')
        self.wv = tf.keras.layers.Dense(d_model, name='dense_v')
        self.dense = tf.keras.layers.Dense(d_model, name='dense_d_model')

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, **kwargs):
        del kwargs
        q, k, v = inputs
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, sequence_length, d_model)
        k = self.wk(k)  # (batch_size, sequence_length, d_model)
        v = self.wv(v)  # (batch_size, sequence_length, d_model)
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v)
        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        output = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        # output = self.dense(output)  # (batch_size, seq_len_q, d_model)

        return output


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_attention_heads, conv_kernel_size, name=None):
        super(EncoderLayer, self).__init__(name=name or 'encoder_layer')
        self._conv = tf.keras.layers.Conv1D(d_model, conv_kernel_size, padding='same')
        self._rnn = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(d_model, return_sequences=True),
            merge_mode='sum', name='rnn')
        self._attention = MultiHeadAttention(d_model, num_attention_heads, name='attention')
        self._layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs, **kwargs):
        """
        (batch_size, seq_len_old, hidden_size_old) -> (batch_size, seq_len_old / pooling_size, hidden_size)
        :param inputs:
        :param kwargs: Keyword arguments for format compatibility.
        :return:
        """
        del kwargs
        q = k = self._rnn(inputs)
        v = self._conv(inputs)
        attention_output = self._attention((q, k, v))
        outputs = self._layer_norm(attention_output + inputs)
        return outputs


class SequenceEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 num_layers, d_model,
                 num_attention_heads,
                 conv_kernel_size,
                 numeric_normalizer_fn=None,
                 name=None):
        super(SequenceEncoder, self).__init__(name=name or 'sequence_encoder')
        self._numeric_normalizer = layers.FunctionWrapper(
            tf.identity if numeric_normalizer_fn is None else numeric_normalizer_fn,
            name='numeric_normalizer')
        self._diff = self._diff = layers.FunctionWrapper(
            functools.partial(diff.diff_1_pad, axis=-2, padding_value=None), name='diff')
        self._sequence_concatenate = tf.keras.layers.Concatenate(axis=-1, name='sequence_concatenate')
        self._sequence_embedding = tf.keras.layers.Dense(d_model, name='sequence_embedding')
        self._mask_embedding = tf.keras.layers.Dense(d_model, name='mask_embedding')
        self._embedding_concatenate = tf.keras.layers.Concatenate(axis=-1, name='embedding_concatenate')
        self._dense = tf.keras.layers.Dense(d_model, name='dense')
        encoder_layers = []
        for i in range(num_layers):
            encoder_layers.append(
                EncoderLayer(d_model, num_attention_heads, conv_kernel_size, name='layer_{}'.format(i)))
        self._encoder_layers = tf.keras.Sequential(encoder_layers, name='encoder_layers')

    def call(self, inputs, mask=None, **kwargs):
        """

        :param inputs:
        :param mask: 0 for mask, 1 for not mask.
        :param kwargs:
        :return:
        """
        del kwargs
        if mask is None:
            mask = tf.ones_like(inputs, dtype=inputs.dtype)
        else:
            inputs = tf.where(tf.cast(mask, tf.bool), inputs, tf.zeros_like(inputs, dtype=inputs.dtype))
            mask = tf.cast(mask, dtype=inputs.dtype)

        if self._numeric_normalizer is not None:
            inputs = self._numeric_normalizer(inputs)
        diff_inputs = self._diff(inputs)
        sequence_inputs = self._sequence_concatenate([inputs, diff_inputs])
        sequence_embedding = self._sequence_embedding(sequence_inputs)
        mask_embedding = self._mask_embedding(mask)
        encoder_layer_inputs = self._embedding_concatenate([sequence_embedding, mask_embedding])
        encoder_layer_inputs = self._dense(encoder_layer_inputs)
        outputs = self._encoder_layers(encoder_layer_inputs)
        return outputs
