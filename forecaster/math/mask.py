""" Mask OPs. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import collections
import tensorflow as tf


class RandomMaskSpec(collections.namedtuple(
    'RandomMaskSpec', (
        'min_mask_len', 'max_mask_len', 'dtype'))):
    @property
    @abc.abstractmethod
    def mask_ratio(self):
        raise NotImplementedError('RandomMaskSpec.mask_ratio')


class GaussSequenceMaskSpec(RandomMaskSpec):
    def __new__(cls, mask_ratio_mean, mask_ratio_stddev,
                min_mask_len=None, max_mask_len=None,
                dtype=tf.bool):
        instance = super(GaussSequenceMaskSpec, cls).__new__(
            cls, min_mask_len, max_mask_len, dtype)
        instance.mask_ratio_mean = mask_ratio_mean
        instance.mask_ratio_stddev = mask_ratio_stddev
        return instance

    @property
    def mask_ratio(self):
        return tf.random.normal((), mean=self.mask_ratio_mean, stddev=self.mask_ratio_stddev)


class UniformSequenceMaskSpec(RandomMaskSpec):
    def __new__(cls, mask_ratio_min, mask_ratio_max,
                min_mask_len=None, max_mask_len=None,
                dtype=tf.bool):
        instance = super(UniformSequenceMaskSpec, cls).__new__(
            cls, min_mask_len, max_mask_len, dtype)
        instance.mask_ratio_min = mask_ratio_min
        instance.mask_ratio_max = mask_ratio_max
        return instance

    @property
    def mask_ratio(self):
        return tf.random.uniform((), minval=self.mask_ratio_min, maxval=self.mask_ratio_max)


def _partial_mask(seq_len, mask_spec, fore=False, name=None):
    with tf.name_scope(name or 'partial_mask'):
        seq_len = tf.convert_to_tensor(seq_len, dtype=tf.int32)
        seq_len_float = tf.cast(seq_len, dtype=tf.float32)
        min_mask_len = tf.convert_to_tensor(mask_spec.min_mask_len or 1, dtype=tf.float32)
        max_mask_len = tf.convert_to_tensor(mask_spec.max_mask_len or seq_len_float - 1, dtype=tf.float32)

        mask_len_float = tf.clip_by_value(mask_spec.mask_ratio * seq_len_float, min_mask_len, max_mask_len)
        mask_len = tf.cast(tf.round(mask_len_float), tf.int32)

        mask_sequence = tf.where(
            tf.equal(tf.range(seq_len) < mask_len, fore),
            x=tf.zeros((seq_len,), dtype=mask_spec.dtype),
            y=tf.ones((seq_len,), dtype=mask_spec.dtype))
    return mask_sequence


def fore_sequence_mask(seq_len, mask_spec, name=None):
    return _partial_mask(seq_len, mask_spec, fore=True, name=name or 'fore_mask')


def post_sequence_mask(seq_len, mask_spec, name=None):
    return _partial_mask(seq_len, mask_spec, fore=False, name=name or 'post_mask')


def middle_sequence_mask(seq_len, mask_spec, mask_start_pos=None, name=None):
    with tf.name_scope(name or 'middle_sequence_mask'):
        seq_len = tf.convert_to_tensor(seq_len, dtype=tf.int32)
        seq_len_float = tf.cast(seq_len, dtype=tf.float32)
        min_mask_len = tf.convert_to_tensor(mask_spec.min_mask_len or 1, dtype=tf.float32)
        max_mask_len = tf.convert_to_tensor(mask_spec.max_mask_len or seq_len_float - 1, dtype=tf.float32)

        mask_len_float = tf.clip_by_value(mask_spec.mask_ratio * seq_len_float, min_mask_len, max_mask_len)
        mask_len = tf.cast(tf.round(mask_len_float), tf.int32)

        unmasked = tf.range(seq_len)
        mask_start_pos = mask_start_pos or 1
        if type(mask_start_pos) is not int:
            mask_start_pos = tf.random.uniform(
                (), minval=mask_start_pos[0], maxval=mask_start_pos[1], dtype=tf.int32)
        mask_end_pos = mask_start_pos + mask_len
        mask_sequence = tf.where(
            tf.logical_and(
                tf.greater_equal(unmasked, mask_start_pos),
                tf.less(unmasked, mask_end_pos)),
            tf.zeros_like(unmasked, dtype=mask_spec.dtype),
            tf.ones_like(unmasked, dtype=mask_spec.dtype))
    return mask_sequence


def random_tensor_mask(shape, axis, seq_mask_fn, mask_spec, name=None):
    """ Create mask along `axis` dimension of `tensor` shaped (d_0, d_1, ..., d_axis, ...).
        Arguments:
            shape: A `TensorShape` like, shape of tensor to be masked.
            axis: An `int` scalar `Tensor` like, axis to apply the mask.
            seq_mask_fn: A function, representing teh method to create a sequence mask. Follow
                the signature:
                    Arguments
                        seq_len: An `int` scalar `Tensor` like, sequence length.
                        mask_spec: An instance of `RandomMaskSpec`.
                    Returns
                        A 1-D `Tensor`, representing the sequence mask.
            mask_spec: A mask spec that feed into `seq_mask_fn` as an argument.
            name: A `str`, OP name, defaults to "random_tensor_mask".
    """
    with tf.name_scope(name or 'random_tensor_mask'):
        tensor_shape = tf.convert_to_tensor(shape)
        axis = tf.cond(
            tf.less(axis, 0),
            lambda: tf.shape(tensor_shape)[0] + axis,
            lambda: axis)
        unmasked_shape = tensor_shape[:axis]
        mask_seq_len = tensor_shape[axis]
        mask_shape = tensor_shape[:axis + 1]
        mask_array_size = tf.reduce_prod(unmasked_shape)

        _, mask_array = tf.while_loop(
            lambda step, _: step < mask_array_size,
            lambda step, array: (step + 1, array.write(step, seq_mask_fn(mask_seq_len, mask_spec))),
            [tf.constant(0), tf.TensorArray(mask_spec.dtype, size=mask_array_size, name='mask_array')])
        mask_sequence = tf.reshape(mask_array.stack(), mask_shape)
    return mask_sequence
