""" Numerical difference OPs. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from forecaster.math import array
from utils import typing as os_typing


def diff_1(value, axis=0, name=None):
    """ Compute 1st-order numerical difference.
        Arguments:
            value: A `Tensor`.
            axis: An `int`, axis to compute difference along.
            name: A `str`, OP name, defaults to "diff1"
        Returns:
            A `Tensor` with the same shape with `value` except at axis `axis`.
    """
    with tf.name_scope(name or 'diff1'):
        value = tf.convert_to_tensor(value)
        a = array.slice_from_axis(value, end=-1, axis=axis)
        b = array.slice_from_axis(value, begin=1, axis=axis)
        diff_tensor = b - a
    return diff_tensor


def diff_n(value, orders, axis=0, name=None):
    """ Compute high-order numerical differences.
           Arguments:
               value: A `Tensor`.
               orders: A sequence of `int`, difference orders to be computed.
               axis: An `int`, axis to compute difference along.
               name: A `str`, OP name, defaults to "diffn"
           Returns:
               A `list` of difference `Tensor`s, corresponding to `orders`.
    """
    orders = os_typing.normalize_list_of_type(orders, int)
    diff_tensors_list = [None] * len(orders)
    with tf.name_scope(name or 'diffn'):
        for order in range(1, max(orders) + 1):
            value = diff_1(value, axis=axis)
            if order in orders:
                diff_tensors_list[orders.index(order)] = value
    diff_tensors_list = list(filter(lambda t: t is not None, diff_tensors_list))
    return diff_tensors_list


def diff_pad(value, orders, axis=0, padding_value=None, group_axis=None, name=None):
    """ Compute numerical differences and pad tensors to the same shape.
        Arguments:
            value: A `Tensor`.
            orders: A sequence of `int`, difference orders to be computed.
            axis: An `int`, axis to compute difference along.
            padding_value: A scalar `Tensor` like, padding value at the begginging.
                Defaults to use the first non-empty value of the tensor.
            group_axis: An `int`, tensor dimension to stack difference results when
                grouping. Defaults to not group.
            name: A `str`, OP name, defaults to "diff_pad"
        Returns:
            A `list` of `Tensor` if `group_axis` is `None`, or else, a `Tensor`.
    """
    padded_tensors = []
    with tf.name_scope(name or 'diff_pad'):
        diff_tensors = diff_n(value, orders, axis=axis)
        for i in range(len(orders)):
            order = orders[i]
            diff_tensor = diff_tensors[i]
            if padding_value is None:
                padding_tensor = tf.stack(
                    [array.slice_from_axis(diff_tensor, begin=0, axis=axis, shrink=True) for _ in  range(order)],
                    axis=axis)
            else:
                padding_tensor = tf.fill(
                    tf.shape(array.slice_from_axis(value, begin=0, axis=axis, end=order)),
                    tf.cast(padding_value, value.dtype))
            padded_tensor = tf.concat([padding_tensor, diff_tensor], axis=axis)
            padded_tensors.append(padded_tensor)
        if group_axis is not None:
            padded_tensors = tf.stack(padded_tensors, axis=group_axis)
    return padded_tensors


def diff_1_pad(value, axis=0, padding_value=None, name=None):
    """ Compute 1st-order numerical differences and pad.
        Arguments:
            value: A `Tensor`.
            axis: An `int`, axis to compute difference along.
            padding_value: A scalar `Tensor` like, padding value at the begginging.
                Defaults to use the first non-empty value of the tensor.
            name: A `str`, OP name, defaults to "diff1_pad"
        Returns:
            A `Tensor`.
    """
    return diff_pad(
        value, [1], axis=axis, padding_value=padding_value, group_axis=False, name=name or 'diff1_pad')[0]
