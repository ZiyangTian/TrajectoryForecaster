""" Mask OPs. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf


class SequenceMaskGenerator(object):
    def __init__(self, sequence_length, min_val=None, max_val=None, dtype=tf.bool, name=None):
        self._sequence_length = tf.convert_to_tensor(sequence_length, dtype=tf.int32)
        self._min_val, self._max_val = self._validate_min_max_vals(min_val, max_val)
        self._dtype = dtype
        self._name = name

    def _validate_min_max_vals(self, min_val, max_val):
        min_val = 0 if min_val is None else min_val
        max_val = self._sequence_length + 1 if max_val is None else max_val
        tf.debugging.assert_non_negative(min_val)
        tf.debugging.assert_less(min_val, max_val)
        tf.debugging.assert_less_equal(max_val, self._sequence_length + 1)
        return min_val, max_val

    def _get_mask_size(self):
        return tf.random.uniform((), self._min_val, self._max_val, dtype=tf.int32)

    def generate(self, scatter_mode=False):
        with tf.name_scope(self._name or 'generate_sequence_mask'):
            mask_size = self._get_mask_size()
            mask_start_pos = tf.random.uniform((), 0, self._sequence_length - mask_size + 1, dtype=tf.int32)
            mask_end_pos = mask_start_pos + mask_size

            unmasked = tf.range(self._sequence_length)
            unmasked = tf.cond(
                tf.convert_to_tensor(scatter_mode, tf.bool),
                lambda: tf.random.shuffle(unmasked),
                lambda: unmasked)
            mask_sequence = tf.where(
                tf.logical_and(
                    tf.greater_equal(unmasked, mask_start_pos),
                    tf.less(unmasked, mask_end_pos)),
                tf.zeros_like(unmasked, dtype=self._dtype),
                tf.ones_like(unmasked, dtype=self._dtype))
        return mask_sequence


def sequence_mask_along_axis(shape, axis,
                             min_mask_length=None,
                             max_mask_length=None,
                             dtype=tf.bool,
                             scatter_mode=False,
                             name=None):
    with tf.name_scope(name or 'sequence_mask_along_axis'):
        tensor_shape = tf.convert_to_tensor(shape, dtype=tf.int32)
        tensor_rank = tf.size(tensor_shape)
        axis = tf.convert_to_tensor(axis, dtype=tf.int32)
        axis = tf.where(tf.less(axis, 0), tensor_rank + axis, axis)  # Use `tf.where` instead of `tf.cond` to
                                                                     # avoid unknown error in using `tf.cond`.
        # axis = tf.cond(
        #     tf.less(axis, 0),
        #     lambda: tensor_rank + axis,
        #     lambda: axis)
        sequence_length = tensor_shape[axis]
        generator = SequenceMaskGenerator(
            sequence_length, min_val=min_mask_length, max_val=max_mask_length, dtype=dtype)

        unmasked_shape = tensor_shape[:axis]
        masking_shape = tensor_shape[axis: axis + 1]
        masked_shape = tensor_shape[axis + 1:]
        mask_array_size = tf.reduce_prod(unmasked_shape)

        _, mask_array = tf.while_loop(
            lambda step, _: step < mask_array_size,
            lambda step, array: (
                step + 1,
                array.write(
                    step,
                    tf.broadcast_to(
                        generator.generate(scatter_mode=scatter_mode),
                        tf.concat([masked_shape, masking_shape], 0)))),
            [tf.constant(0), tf.TensorArray(dtype, size=mask_array_size, name='mask_array')])

        mask_tensor = tf.transpose(
            tf.reshape(mask_array.stack(), tf.concat([unmasked_shape, masked_shape, masking_shape], 0)),
            perm=tf.concat([tf.range(0, axis), [tensor_rank - 1], tf.range(axis, tensor_rank - 1)], 0))
        mask_tensor = tf.reshape(mask_tensor, tensor_shape)
        mask_tensor.set_shape(shape)
    return mask_tensor
