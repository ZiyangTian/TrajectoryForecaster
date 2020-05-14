import collections
import os
import tensorflow as tf

from forecaster.apps import base
from forecaster.data import sequence
from forecaster.ops import mask
from forecaster.models import networks
from forecaster.models import optimizers


class Masker(collections.namedtuple(
    'Masker', (
        'sequence_mask_prop',
        'feature_mask_prop',
        'min_sequence_mask_length',
        'max_sequence_mask_length',
        'min_feature_mask_length',
        'max_feature_mask_length',
        'use_scatter_mask_at_sequence_in_prop',
        'use_scatter_mask_at_feature_in_prop',
        'dtype'))):
    def __new__(cls,
                sequence_mask_prop,
                feature_mask_prop,
                min_sequence_mask_length,
                max_sequence_mask_length,
                min_feature_mask_length,
                max_feature_mask_length,
                use_scatter_mask_at_sequence_in_prop=0.,
                use_scatter_mask_at_feature_in_prop=0.,
                dtype=tf.int32):
        if not(sequence_mask_prop > 0 and feature_mask_prop > 0 and sequence_mask_prop + feature_mask_prop == 1):
            raise ValueError('Invalid values of `sequence_mask_prop` and `sequence_mask_prop`.')
        if not 0 < min_sequence_mask_length < max_sequence_mask_length:
            raise ValueError('Invalid values of `min_sequence_mask_length` and `max_sequence_mask_length`.')
        if not 0 < min_feature_mask_length < max_feature_mask_length:
            raise ValueError('Invalid values of `min_feature_mask_length` and `max_feature_mask_length`.')
        if not 0 < use_scatter_mask_at_sequence_in_prop < 1:
            raise ValueError('`use_scatter_mask_at_sequence_in_prop` must be ranged 0 to 1, got {} instead.'
                             .format(use_scatter_mask_at_sequence_in_prop))
        if not 0 < use_scatter_mask_at_feature_in_prop < 1:
            raise ValueError('`use_scatter_mask_at_feature_in_prop` must be ranged 0 to 1, got {} instead.'
                             .format(use_scatter_mask_at_feature_in_prop))
        return super(Masker, cls).__new__(
            cls,
            sequence_mask_prop,
            feature_mask_prop,
            min_sequence_mask_length,
            max_sequence_mask_length,
            min_feature_mask_length,
            max_feature_mask_length,
            use_scatter_mask_at_sequence_in_prop,
            use_scatter_mask_at_feature_in_prop,
            dtype)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def _generate_sequence_mask(self, shape, scatter_mode=False):
        return mask.sequence_mask_along_axis(
            shape, 0,
            min_mask_length=self.min_sequence_mask_length,
            max_mask_length=self.max_sequence_mask_length,
            dtype=tf.int32,
            scatter_mode=scatter_mode)

    def _generate_feature_mask(self, shape, scatter_mode=False):
        mask_tensor = mask.sequence_mask_along_axis(
            (shape[1], shape[0]), -1,
            min_mask_length=self.min_feature_mask_length,
            max_mask_length=self.max_feature_mask_length,
            dtype=tf.int32,
            scatter_mode=scatter_mode)
        mask_tensor = tf.transpose(mask_tensor)
        return mask_tensor

    def _sequence_mask(self, shape):
        return tf.cond(
            tf.less(tf.random.uniform((), 0, 1), self.use_scatter_mask_at_sequence_in_prop),
            lambda: self._generate_sequence_mask(shape, scatter_mode=True),
            lambda: self._generate_sequence_mask(shape, scatter_mode=False))

    def _feature_mask(self, shape):
        return tf.cond(
            tf.less(tf.random.uniform((), 0, 1), self.use_scatter_mask_at_feature_in_prop),
            lambda: self._generate_feature_mask(shape, scatter_mode=True),
            lambda: self._generate_feature_mask(shape, scatter_mode=False))

    def generate_mask(self, shape, name=None):
        with tf.name_scope(name or 'generate_mask'):
            mask_tensor = tf.cond(
                tf.less(tf.random.uniform((), 0, 1), self.sequence_mask_prop),
                lambda: self._sequence_mask(shape),
                lambda: self._feature_mask(shape))
        return mask_tensor
