""" Array OPs. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf


def slice_assigned(tensor, value,
                   begin, end, strides=None,
                   begin_mask=0, end_mask=0,
                   ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0,
                   name=None):
    """Create a tensor by assigning values to a tensor slice.
    Arguments:
        tensor: A `Tensor` like, the tensor to be assigned value to.
        value: A `Tensor` like, the value to assign. Must has the same shape and data type
            with the tensor slice.
        begin: See `tf.strided_slice`.
        end:  See `tf.strided_slice`.
        strides:  See `tf.strided_slice`.
        begin_mask:  See `tf.strided_slice`.
        end_mask:  See `tf.strided_slice`.
        ellipsis_mask:  See `tf.strided_slice`.
        new_axis_mask:  See `tf.strided_slice`.
        shrink_axis_mask:  See `tf.strided_slice`.
        name: A `str`, `OP` name.
    Returns:
        A `Tensor` with the same shape with `tensor`. The value of `tensor` will not change.
    """
    with tf.name_scope(name or 'slice_assigned'):
        tensor = tf.convert_to_tensor(tensor)
        tensor_shape = tf.shape(tensor)
        tensor_size = tf.size(tensor)

        index_slice = tf.strided_slice(
            tf.reshape(tf.range(tensor_size,), tensor_shape),
            begin, end, strides=strides,
            begin_mask=begin_mask, end_mask=end_mask,
            ellipsis_mask=ellipsis_mask, new_axis_mask=new_axis_mask,
            shrink_axis_mask=shrink_axis_mask)
        tf.assert_equal(tf.shape(index_slice), tf.shape(value))
        flattened_index_slice = tf.expand_dims(tf.keras.backend.flatten(index_slice), -1)
        flattened_value = tf.keras.backend.flatten(value)
        scattered_value = tf.scatter_nd(flattened_index_slice, flattened_value, (tensor_size,))
        padded_value = tf.reshape(scattered_value, tensor_shape)

        flattened_ones = tf.ones_like(flattened_value, dtype=tf.bool)
        scattered_ones = tf.scatter_nd(flattened_index_slice, flattened_ones, (tensor_size,))
        padded_ones = tf.reshape(scattered_ones, tensor_shape)

        assigned_tensor = tf.where(padded_ones, padded_value, tensor)
    return assigned_tensor


def element_assigned(tensor, value, position, name=None):
    """Create a tensor by assigning values to an element.
    Arguments:
        tensor: A `Tensor` like, the tensor to be assigned value to.
        value: A `Tensor` scalar like, the value to assign. Must has the same data type with `tensor`.
        position: A 1-D `int` `Tensor` like, representing the element position to assign.
        name: A `str`, `OP` name.
    Returns:
        A `Tensor` with the same shape with `tensor`. The value of `tensor` will not change.
    """
    with tf.name_scope(name or 'element_assigned'):
        position = tf.convert_to_tensor(position)
        begin = position
        end = position + tf.ones_like(position)
        shrink_axis_mask = tf.bitwise.left_shift(1, tf.rank(tensor) + 1) - 1
        assigned_tensor = slice_assigned(tensor, value, begin, end, shrink_axis_mask=shrink_axis_mask)
    return assigned_tensor


def slice_from_axis(value, begin=None, end=None, stride=None,
                    axis=0, shrink=False, name=None):
    """Slice a tensor along a specified axis.
    Arguments:
        value: An N-D `Tensor` with a known rank.
        begin: An `int` scalar `Tensor` like, slice beginning position. Defaults to slice
            from the head.
        end: An `int` scalar `Tensor` like, slice ending position. Defaults to slice
            until the tail.
        stride: An `int` scalar `Tensor` like, axis to slice along.
        axis: An `int` scalar `Tensor` like, axis to slice along.
        shrink: A `bool` scalar `Tensor` like, whether to squeeze the dimension.
        name: A `str`, OP name, defaults to "slice_from_axis".
    Returns:
        A sliced `Tensor`.
    """
    with tf.name_scope(name or 'slice_from_axis'):
        value = tf.convert_to_tensor(value)
        rank = tf.rank(value)
        axis = tf.cond(
            axis < 0,
            lambda: rank + axis,
            lambda: axis)

        all_mask = tf.bitwise.left_shift(1, rank) - 1
        spec_mask = tf.bitwise.bitwise_xor(all_mask, tf.bitwise.left_shift(1, axis))

        begins = tf.zeros((rank,), dtype=tf.int32)
        if begin is None:
            begin_mask = all_mask
        else:
            begins = element_assigned(begins, begin, (axis,))
            begin_mask = spec_mask

        ends = tf.zeros([rank], dtype=tf.int32)
        if end is None:
            end_mask = all_mask
        else:
            ends = element_assigned(ends, end, (axis,))
            end_mask = spec_mask

        strides = tf.ones((rank,), dtype=tf.int32)
        if stride is not None:
            strides = element_assigned(strides, stride, (axis,))
        shrink_mask = tf.cond(
            shrink,
            lambda: tf.bitwise.invert(spec_mask),
            lambda: tf.bitwise.invert(all_mask))

    return tf.strided_slice(
        value, begins, ends,
        strides=strides, begin_mask=begin_mask, end_mask=end_mask,
        shrink_axis_mask=shrink_mask, name=name or 'slice_from_axis')


def stack_n(value, n, axis=0, name=None):
    """Stack `n` uniform tensors into one, like `tf.stack`.
    Arguments:
        value: A `Tensor` like, value to stack.
        n: An `int` scalar `Tensor` like, number of values to stack.
        axis: An `int` scalar `Tensor` like, stack axis.
        name: A `str`, OP name, defaults to "stack_n".
    Returns:
        A stacked `Tensor`.
    """
    with tf.name_scope(name or 'stack_n'):
        value = tf.expand_dims(value, axis=axis)
        stacked = tf.gather(value, tf.zeros((n,), dtype=tf.int32), axis=axis)
    return stacked
