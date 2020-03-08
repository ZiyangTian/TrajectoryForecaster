import tensorflow as tf

from forecaster.model import layers


def scaled_dot_product_attention(q, k, v, mask=None, name=None):
    """ Compute attention weights.
        Arguments
            q: query, shape == (..., seq_len_q, depth)
            k: key, shape == (..., seq_len, depth)
            v: value, shape == (..., seq_len, depth_v)
            mask: A float Tensor with shape broadcastable to
                  (..., seq_len_q, seq_len). Defaults to None.
        Returns
            Outputs. Attention weights.
        * q, k, v must have must have matching leading dimensions.
    """
    with tf.name_scope(name or 'scaled_dot_product_attention'):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)  # (..., seq_len_q, seq_len)

        if mask is not None:
            scaled_attention_logits += (1 - tf.cast(mask, tf.float32)) * 1e9

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_output = tf.matmul(attention_weights, v)
    return attention_output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_size, pooling_size, kernel_size, name=None):
        super(EncoderLayer, self).__init__(name=name or 'encoder_layer')
        self._conv = tf.keras.layers.Conv1D(hidden_size, kernel_size, padding='same')
        self._pooling = tf.keras.layers.AveragePooling1D(pooling_size, padding='same')
        self._rnn = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_size, return_sequences=True),
            merge_mode='sum', name='rnn')
        self._layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs, training=None, mask=None):
        """
        (batch_size, seq_len_old, hidden_size_old) -> (batch_size, seq_len_old / pooling_size, hidden_size)
        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        del training
        q = self._pooling(self._conv(inputs))
        k = v = self._rnn(inputs)
        attention_output, _ = scaled_dot_product_attention(q, k, v, mask=mask)
        outputs = self._layer_norm(attention_output)
        return outputs


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_size, stride, kernel_size, name=None):
        super(DecoderLayer, self).__init__(name=name or 'decoder_layer')
        self._hidden_size = hidden_size
        self._stride = stride
        self._kernel_size = kernel_size
        self._kernel = None
        self._conv1d_transpose_output_shape = None

        self._rnn = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_size, return_sequences=True),
            merge_mode='sum', name='rnn')
        self._layer_norm = tf.keras.layers.LayerNormalization()

    def build(self, input_shape):
        self._kernel = self.add_weight(
            name='conv1d_transpose_kernal',
            shape=(self._kernel_size, self._hidden_size, input_shape[-1]),
            dtype=tf.float32)
        self._conv1d_transpose_output_shape = tf.constant(
            (input_shape[0], input_shape[1] * self._stride, self._hidden_size))

    def call(self, inputs, training=None, mask=None):
        """
        (batch_size, seq_len_old, hidden_size_old) -> (batch_size, seq_len_old * stride, hidden_size)
        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        del training
        del mask
        q = tf.nn.conv1d_transpose(inputs, self._kernel, self._conv1d_transpose_output_shape, self._stride)
        k = v = self._rnn(inputs)
        attention_output, _ = scaled_dot_product_attention(q, k, v)
        outputs = self._layer_norm(attention_output)
        return outputs


class Encoder(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_sizes, pooling_sizes, kernel_sizes,
                 name=None):
        super(Encoder, self).__init__(name=name or 'encoder')
        self._layers = []
        for i, (hidden_size, pooling_size, kernel_size) in enumerate(
                zip(hidden_sizes, pooling_sizes, kernel_sizes)):
            self._layers.append(EncoderLayer(hidden_size, pooling_size, kernel_size, name='layer_{}'.format(i)))
        self._flatten = tf.keras.layers.Flatten()

    def call(self, inputs, training=None, mask=None):
        encoder_layer_inputs = inputs
        for layer in self._layers:
            encoder_layer_inputs = layer(encoder_layer_inputs, training=training, mask=mask)
        encoder_outputs = self._flatten(encoder_layer_inputs)
        return encoder_outputs


class Decoder(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_sizes, strides, kernel_sizes,
                 name=None):
        super(Decoder, self).__init__(name=name or 'decoder')
        self._layers = []
        for i, (hidden_size, stride, kernel_size) in enumerate(
                zip(hidden_sizes, strides, kernel_sizes)):
            self._layers.append(DecoderLayer(hidden_size, stride, kernel_size, name='layer_{}'.format(i)))

    def call(self, inputs, training=None, mask=None):
        decoder_layer_inputs = tf.expand_dims(inputs, axis=1)
        for layer in self._layers:
            decoder_layer_inputs = layer(decoder_layer_inputs, training=training, mask=mask)
        decoder_outputs = decoder_layer_inputs
        return decoder_outputs


class SequenceSelfEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 encoder_hidden_sizes,
                 encoder_pooling_sizes,
                 encoder_kernel_sizes,
                 decoder_hidden_sizes,
                 decoder_strides,
                 decoder_kernel_sizes,
                 uniform_centre=None,
                 uniform_bound=None,
                 name=None):
        super(SequenceSelfEncoder, self).__init__(name=name or 'sequence_self_encoder')
        self._uniform_diff = layers.UniformDiff(uniform_centre, uniform_bound, diff_axis=1)
        self._encoder = Encoder(encoder_hidden_sizes, encoder_pooling_sizes, encoder_kernel_sizes)
        self._decoder = Decoder(decoder_hidden_sizes, decoder_strides, decoder_kernel_sizes)
        self._restore = layers.Restore(uniform_centre, uniform_bound)

    @staticmethod
    def _apply_mask(inputs, mask=None):
        if mask is None:
            return inputs
        return inputs  # TODO:...

    def _encode(self, inputs, training=None, mask=None):
        inputs = self._apply_mask(inputs, mask=mask)
        return self._encoder(self._uniform_diff(inputs), training=training, mask=mask)

    def _decode(self, inputs, training=None, mask=None):
        del training
        del mask
        outputs = self._decoder(inputs, training=training, mask=mask)
        outputs = self._restore(outputs)
        return outputs

    def call(self, inputs, training=None, mask=None, mode=None):
        if mode is None:
            return self._decode(self._encode(inputs, training=training, mask=mask))
        if mode is 'encode':
            return self._encode(inputs, training=training, mask=mask)
        if mode is 'decode':
            return self._decode(inputs, training=training, mask=mask)

    def encode(self, inputs, training=None, mask=None):
        return self.__call__(inputs, training=training, mask=mask, mode='encode')

    def decode(self, inputs, training=None, mask=None):
        return self.__call__(inputs, training=training, mask=mask, mode='decode')


x = tf.constant(1., shape=(3, 40, 14))
encoder = Encoder([32, 64], [4, 10], [5, 5],)
outputs = encoder(x)
print(outputs.shape)
decoder = Decoder([32, 7], [10, 2], [5, 5])
outputs = decoder(outputs)
print(outputs.shape)
# layer = DecoderLayer(20, 3, 5)
# a = tf.constant(1., shape=(2, 1, 100))
# b = layer(a)
# layer2 = DecoderLayer(40, 3, 5)
# c = layer2(b)
# print(c.shape)
# filters = tf.constant(1., shape=(3, 50, 100))
# b = tf.nn.conv1d_transpose(a, filters, (2, 2, 50), 2)



