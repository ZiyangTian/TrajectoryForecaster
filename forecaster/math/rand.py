""" . """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf


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
