""" Array OPs. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf


def slice_from_axis(value, begin=None, end=None, stride=None,
                    axis=0, shrink=False, name=None):
    """ Slice a tensor along a specified axis.
        Arguments:
            value: An N-D `Tensor` with a known rank.
            begin: An `int` scalar `Tensor` like, slice beginning position. Defaults to slice
                from the head.
            end: An `int` scalar `Tensor` like, slice ending position. Defaults to slice
                until the tail.
            stride: An `int`, axis to slice along.
            axis: An `int`, axis to slice along.
            shrink: A `bool`, whether to squeeze the dimension.
            name: A `str`, OP name, defaults to "slice_from_axis".
        Returns:
            A sliced `Tensor`.
    """
    value = tf.convert_to_tensor(value)
    dim = len(value.shape)
    if axis < 0:
        axis = dim + axis

    all_mask = (1 << dim) - 1
    spec_mask = all_mask ^ 1 << axis

    begins = [0] * dim
    if begin is None:
        begin_mask = all_mask
    else:
        begins[axis] = begin
        begin_mask = spec_mask

    ends = [0] * dim
    if end is None:
        end_mask = all_mask
    else:
        ends[axis] = end
        end_mask = spec_mask

    strides = [1] * dim
    if stride is not None:
        strides[axis] = stride

    shrink_mask = ~all_mask
    if shrink:
        shrink_mask = ~spec_mask

    return tf.strided_slice(
        value, begins, ends,
        strides=strides, begin_mask=begin_mask, end_mask=end_mask,
        shrink_axis_mask=shrink_mask, name=name or 'slice_from_axis')


def stack_n(value, n, axis=0, name=None):
    """ Stack `n` uniform tensors into one, like `tf.stack`.
        Arguments:
            value: A `Tensor` like, value to stack.
            n: A `tf.int64` scalar `Tensor`, number of values to stack.
            axis: An `int`, stack axis.
            name: A `str`, OP name, defaults to "stack_n".
        Returns:
            A stacked `Tensor`.
    """
    with tf.name_scope(name or 'stack_n'):
        value = tf.expand_dims(value, axis=axis)
        stacked = tf.gather(value, tf.zeros((n,), dtype=tf.int32), axis=axis)
    return stacked
