""" Uniforming and restoring OPs. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf


def denominator_clip(tensor, minimum, name=None):
    """Clip a tensor by a specified minimum absolute value.
        Arguments:
            tensor: A `Tensor` to be clipped.
            minimum: A `Tensor` with the same type as `tensor` and broadcastable to
                the shape of `tensor`
            name: A `str`, OP name, defaults to "denominator_clip".
        Returns:
            A clipped `Tensor`.
    """
    with tf.name_scope(name or 'denominator_clip'):
        tensor = tf.where(
            tensor >= 0,
            tf.where(tensor < minimum, tf.fill(tf.shape(tensor), minimum), tensor),
            tensor)
        tensor = tf.where(
            tensor < 0,
            tf.where(tensor > -minimum, tf.fill(tf.shape(tensor), -minimum), tensor),
            tensor)
    return tensor
