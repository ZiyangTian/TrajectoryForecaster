""" Metrics. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from tensorflow.python.keras.metrics import MeanMetricWrapper


def _square_deviations(y_true, y_pred):
    return tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)


def _deviations(y_true, y_pred):
    return tf.sqrt(_square_deviations(y_true, y_pred))


def _square_destination_deviations(y_true, y_pred):
    return tf.reduce_sum(tf.square(y_true[:, -1, :] - y_pred[:, -1, :]), axis=-1)


def destination_deviation(y_true, y_pred):
    return tf.sqrt(_square_destination_deviations(y_true, y_pred))


def max_deviation(y_true, y_pred):
    return tf.reduce_max(_deviations(y_true, y_pred), axis=-1)


def min_deviation(y_true, y_pred):
    return tf.reduce_min(_deviations(y_true, y_pred), axis=-1)


def mean_deviation(y_true, y_pred):
    return tf.reduce_mean(_deviations(y_true, y_pred), axis=-1)


class MeanDeviation(MeanMetricWrapper):
    def __init__(self, name='mean_deviation', dtype=None):
        super(MeanDeviation, self).__init__(mean_deviation, name, dtype=dtype)


class MaxDeviation(MeanMetricWrapper):
    def __init__(self, name='max_deviation', dtype=None):
        super(MaxDeviation, self).__init__(max_deviation, name, dtype=dtype)


class MinDeviation(MeanMetricWrapper):
    def __init__(self, name='min_deviation', dtype=None):
        super(MinDeviation, self).__init__(min_deviation, name, dtype=dtype)


class DestinationDeviation(MeanMetricWrapper):
    def __init__(self, name='destination_deviation', dtype=None):
        super(DestinationDeviation, self).__init__(destination_deviation, name, dtype=dtype)


def get_keras_metrics(*string_identifies):
    _metrics = {
        'max_deviation': MaxDeviation,
        'mean_deviation': MeanDeviation(),
        'min_deviation': MinDeviation,
        'destination_deviation': DestinationDeviation}
    return list(map(lambda k: _metrics[k], string_identifies))
