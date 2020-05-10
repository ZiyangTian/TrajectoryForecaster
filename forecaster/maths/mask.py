""" Mask OPs. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf


class SequenceMaskGenerator(object):
    def __init__(self, sequence_length, min_val=None, max_val=None, dtype=tf.bool):
        self._sequence_length = tf.convert_to_tensor(sequence_length, dtype=tf.int32)
        self._min_val = 0 if min_val is None else min_val
        self._max_val = sequence_length + 1 if max_val is None else max_val
        self._dtype = dtype

    def _get_mask_size(self):
        return tf.random.uniform((), self._min_val, self._max_val, dtype=tf.int32)

    def generate(self, scatter_mode=False, name=None):
        with tf.name_scope(name or 'sequence_mask'):
            mask_size = self._get_mask_size()
            mask_start_pos = tf.random.uniform((), 0, self._sequence_length - mask_size + 1, dtype=tf.int32)
            mask_end_pos = mask_start_pos + mask_size

            unmasked = tf.range(self._sequence_length)
            unmasked = tf.cond(scatter_mode, lambda: tf.random.shuffle(unmasked), lambda: unmasked)
            mask_sequence = tf.where(
                tf.logical_and(
                    tf.greater_equal(unmasked, mask_start_pos),
                    tf.less(unmasked, mask_end_pos)),
                tf.zeros_like(unmasked, dtype=self._dtype),
                tf.ones_like(unmasked, dtype=self._dtype))
        return mask_sequence

    def mask_tensor(self, shape, axis=-1, name=None):
        with tf.name_scope(name or 'mask_tensor'):
            tensor_shape = tf.convert_to_tensor(shape)
            axis = tf.cond(
                tf.less(axis, 0),
                lambda: tf.shape(tensor_shape)[0] + axis,
                lambda: axis)

            unmasked_shape = tensor_shape[:axis]
            mask_seq_len = tensor_shape[axis: axis + 1]
            mask_shape = tensor_shape[axis + 1:]
            mask_array_size = tf.reduce_prod(unmasked_shape)

            _, mask_array = tf.while_loop(
                lambda step, _: step < mask_array_size,
                lambda step, array: (
                    step + 1,
                    array.write(
                        step,
                        tf.broadcast_to(
                            self.generate(),
                            tf.concat([mask_shape, mask_seq_len], 0)))),
                [tf.constant(0), tf.TensorArray(self._dtype, size=mask_array_size, name='mask_array')])
            mask_sequence = mask_array.stack()  # TODO: transpose

        return mask_sequence



g = SequenceMaskGenerator(10, 3, 8, dtype=tf.int32)
print(g.mask_tensor((3,10,4),axis=-2).numpy())


