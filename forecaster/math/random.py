""" . """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import collections
import tensorflow as tf


class RandomScalarSpec(collections.namedtuple(
    'RandomScalarSpec', (
        'min_val', 'max_val', 'dtype'))):
    @property
    @abc.abstractmethod
    def value(self):
        raise NotImplementedError('RandomMaskSpec.mask_ratio')


class GaussSequenceMaskSpec(RandomScalarSpec):
    def __new__(cls, mask_ratio_mean, mask_ratio_stddev,
                min_mask_len=None, max_mask_len=None,
                dtype=tf.bool):
        instance = super(GaussSequenceMaskSpec, cls).__new__(
            cls, min_mask_len, max_mask_len, dtype)
        instance.mask_ratio_mean = mask_ratio_mean
        instance.mask_ratio_stddev = mask_ratio_stddev
        return instance

    @property
    def value(self):
        return tf.random.normal((), mean=self.mask_ratio_mean, stddev=self.mask_ratio_stddev)


# TODO:
def map_between_shuffled(tensor, dtype=None, name=None):
    with tf.name_scope(name or 'map_between_shuffled'):
        seq = tf.range(tf.shape(tensor)[0])
        index = tf.range(tf.shape(tensor)[0])
        seq_and_index = tf.stack([seq, index], axis=-1)
        fst_and_index = tf.random.shuffle(seq_and_index)
        link = tf.stack([fst_and_index[:, 0], seq_and_index[:, 1]], axis=-1)
        shuffled_link = tf.random.shuffle(link)
        fst = fst_and_index[:, 0]
        snd = shuffled_link[:, 0]
        mapping = shuffled_link[:, 1]
    return fst, snd, mapping
