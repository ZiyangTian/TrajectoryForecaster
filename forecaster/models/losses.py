""" Loss functions. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from tensorflow.python.keras.losses import LossFunctionWrapper


def one_hot_with_exception(indicator,
                           num_features=None,
                           exception_value=None,
                           dtype=tf.float32,
                           name=None):
    with tf.name_scope(name or 'one_hot_with_exception'):
        indicator = tf.convert_to_tensor(indicator, tf.int32)
        shape = tf.shape(indicator)
        flattened_indicator = tf.reshape(indicator, [-1])
        flattened_length = tf.shape(flattened_indicator)[0]
        num_features = tf.reduce_max(tf.reduce_max(flattened_indicator)) + 1 if num_features is None else num_features
        flattened_indicator_with_all_dims = tf.transpose(tf.broadcast_to(
            flattened_indicator, (num_features, flattened_length)))
        exception_value = 1. / tf.cast(num_features, dtype) if exception_value is None else exception_value
        flattened_one_hot = tf.where(
            tf.less(flattened_indicator_with_all_dims, 0),
            tf.ones_like(flattened_indicator_with_all_dims, dtype=dtype) * exception_value,
            tf.where(
                tf.equal(flattened_indicator_with_all_dims, tf.broadcast_to(
                    tf.range(num_features), (flattened_length, num_features))),
                tf.ones_like(flattened_indicator_with_all_dims, dtype=dtype),
                tf.zeros_like(flattened_indicator_with_all_dims, dtype=dtype)))
        one_hot = tf.reshape(flattened_one_hot, tf.concat([shape, [-1]], axis=0))
    return one_hot


def sparse_with_exception_softmax_cross_entropy(labels, logits, name=None):
    with tf.name_scope(name or 'sparse_with_exception_softmax_cross_entropy'):
        one_hot_labels = one_hot_with_exception(labels, num_features=tf.shape(logits)[-1])
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=logits)
    return cross_entropy


def normalized_mean_square_error(y_true, y_pred, numeric_normalizer_fn=None):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    if numeric_normalizer_fn is not None:
        y_pred = numeric_normalizer_fn(y_pred)
        y_true = numeric_normalizer_fn(y_true)
    return tf.keras.losses.mean_squared_error(y_true, y_pred)


class NormalizedMeanSquareError(LossFunctionWrapper):
    def __init__(self, numeric_normalizer_fn=None, name=None):
        super(NormalizedMeanSquareError, self).__init__(
            normalized_mean_square_error,
            name=name or 'normalized_mean_square_error',
            numeric_normalizer_fn=numeric_normalizer_fn)
