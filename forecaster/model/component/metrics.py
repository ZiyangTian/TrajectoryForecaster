""" Metrics. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from tensorflow.python.ops.metrics_impl import metric_variable


def _square_deviations_in_batch(labels, predictions):
    return tf.reduce_sum(tf.square(labels - predictions), axis=-1)


def _deviations_in_batch(labels, predictions):
    return tf.sqrt(_square_deviations_in_batch(labels, predictions))


def _square_destination_deviations_in_batch(labels, predictions):
    return tf.reduce_sum(
        tf.square(labels[:, -1, :] - predictions[:, -1, :]),
        axis=-1)


def _destination_deviations_in_batch(labels, predictions):
    return tf.sqrt(_square_destination_deviations_in_batch(labels, predictions))


def maximum(values,
            metrics_collections=None,
            updates_collections=None,
            name=None):
    with tf.variable_scope(name, 'maximum', [values]):
        values = tf.to_float(values)

    max_value = metric_variable([], tf.float32, name='max_value')
    values_max = tf.math.reduce_max(values, axis=None)

    with tf.control_dependencies([values]):
        update_op = tf.math.maximum(max_value, values_max)
        with tf.control_dependencies([update_op]):
            # TODO: Add distributed evaluation support.
            metric_value = update_op

    if metrics_collections:
        tf.add_to_collections(metrics_collections, metric_value)
    if updates_collections:
        tf.add_to_collections(updates_collections, update_op)

    return metric_value, update_op


def minimum(values,
            metrics_collections=None,
            updates_collections=None,
            name=None):
    with tf.variable_scope(name, 'minimum', [values]):
        values = tf.to_float(values)

    min_value = metric_variable([], tf.float32, name='max_value')
    values_min = tf.math.reduce_min(values, axis=None)

    with tf.control_dependencies([values]):
        update_op = tf.cond(  # `min_value` is initiated with zero.
            tf.equal(min_value, 0.),
            true_fn=lambda: values_min,
            false_fn=lambda: tf.math.minimum(min_value, values_min))
        with tf.control_dependencies([update_op]):
            # TODO: Add distributed evaluation support.
            metric_value = update_op

    if metrics_collections:
        tf.add_to_collections(metrics_collections, metric_value)
    if updates_collections:
        tf.add_to_collections(updates_collections, update_op)

    return metric_value, update_op


mean = tf.metrics.mean
mean_square_error = tf.metrics.mean_squared_error
mean_absolute_error = tf.metrics.mean_absolute_error


def mean_deviation(labels,
                   predictions,
                   weights=None,
                   metrics_collections=None,
                   updates_collections=None,
                   name=None):
    with tf.name_scope(name or 'mean_deviation'):
        deviations = _deviations_in_batch(labels, predictions)
        return mean(
            deviations,
            weight=weights,
            metrics_collections=metrics_collections,
            updates_collections=updates_collections)


def max_deviation(labels,
                  predictions,
                  weights=None,
                  metrics_collections=None,
                  updates_collections=None,
                  name=None):
    if weights is not None:
        tf.logging.warn('Argument `weights` is not supported in function `max_deviation`, neglected.')
    del weights
    with tf.name_scope(name or 'max_deviation'):
        deviations = _deviations_in_batch(labels, predictions)
        return maximum(
            deviations,
            metrics_collections=metrics_collections,
            updates_collections=updates_collections)


def min_deviation(labels,
                  predictions,
                  weights=None,
                  metrics_collections=None,
                  updates_collections=None,
                  name=None):
    if weights is not None:
        tf.logging.warn('Argument `weights` is not supported in function `min_deviation`, neglected.')
    del weights
    with tf.name_scope(name or 'min_deviation'):
        deviations = _deviations_in_batch(labels, predictions)
        return minimum(
            deviations,
            metrics_collections=metrics_collections,
            updates_collections=updates_collections)


def mean_destination_deviation(labels,
                               predictions,
                               weights=None,
                               metrics_collections=None,
                               updates_collections=None,
                               name=None):
    with tf.name_scope(name or 'mean_destination_deviation'):
        destination_deviations = _destination_deviations_in_batch(labels, predictions)
        return mean(
            destination_deviations,
            weight=weights,
            metrics_collections=metrics_collections,
            updates_collections=updates_collections)


def max_destination_deviation(labels,
                              predictions,
                              weights=None,
                              metrics_collections=None,
                              updates_collections=None,
                              name=None):
    if weights is not None:
        tf.logging.warn('Argument `weights` is not supported in function '
                        '`max_destination_deviation`, neglected.')
    del weights
    with tf.name_scope(name or 'max_destination_deviation'):
        destination_deviations = _destination_deviations_in_batch(labels, predictions)
        return maximum(
            destination_deviations,
            metrics_collections=metrics_collections,
            updates_collections=updates_collections)


def min_destination_deviation(labels,
                              predictions,
                              weights=None,
                              metrics_collections=None,
                              updates_collections=None,
                              name=None):
    if weights is not None:
        tf.logging.warn('Argument `weights` is not supported in function '
                        '`min_destination_deviation`, neglected.')
    del weights
    with tf.name_scope(name or 'min_destination_deviation'):
        destination_deviations = _destination_deviations_in_batch(labels, predictions)
        return minimum(
            destination_deviations,
            metrics_collections=metrics_collections,
            updates_collections=updates_collections)
