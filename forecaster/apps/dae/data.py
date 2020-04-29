""" . """
import os
import functools
import json
import shutil
import tensorflow as tf

# from forecaster.apps import base as app_base
from forecaster.data import sequence
# from forecaster.math import uniform
# from forecaster.model.component import metrics
# from utils.os import files_in_dir
# from utils.typing import normalize_list_of_type


def uniform_random_sequence_mask(seq_len, min_mask_len=None, max_mask_len=None, dtype=tf.bool, name=None):
    if min_mask_len is None:
        min_mask_len = 1
    if max_mask_len is None:
        max_mask_len = seq_len - 1
    with tf.name_scope(name or 'uniform_random_sequence_mask'):
        mask_len = tf.random.uniform((), minval=min_mask_len, maxval=max_mask_len + 1, dtype=tf.int32)
        start_pos = tf.random.uniform((), minval=0, maxval=seq_len - mask_len + 1, dtype=tf.int32)
        end_pos = start_pos + mask_len
        mask_sequence = tf.cast(
            tf.logical_or(
                tf.less(tf.range(seq_len), tf.broadcast_to(start_pos, (seq_len,))),
                tf.greater_equal(tf.range(seq_len), tf.broadcast_to(end_pos, (seq_len,)))),
            dtype)
    return mask_sequence


def uniform_random_feature_mask(seq_len, num_features,
                                min_mask_len=None, max_mask_len=None,
                                dtype=tf.bool, name=None):
    with tf.name_scope(name or 'uniform_random_feature_mask'):
        _, mask_array = tf.while_loop(
            lambda step, _: step < seq_len,
            lambda step, array: (step + 1, array.write(step, uniform_random_sequence_mask(
                num_features, min_mask_len=min_mask_len, max_mask_len=max_mask_len, dtype=dtype))),
            [tf.constant(0), tf.TensorArray(dtype, size=seq_len)])
        mask_sequences = mask_array.stack()
    return mask_sequences


def _apply_sequence_mask(feature_tensor, min_mask_len, max_mask_len):
    """ feature_tensor: (seq_len, num_features) """
    with tf.name_scope('apply_sequence_mask'):
        shape = tf.shape(feature_tensor)
        mask_sequence = uniform_random_sequence_mask(
            shape[0], min_mask_len=min_mask_len, max_mask_len=max_mask_len)
        full_mask = tf.transpose(tf.broadcast_to(mask_sequence, (shape[1], shape[0])))
        masked_tensor = tf.where(full_mask, feature_tensor, tf.zeros_like(feature_tensor, dtype=feature_tensor.dtype))
    return masked_tensor, full_mask


def _apply_feature_mask(feature_tensor, min_mask_len, max_mask_len):
    """ feature_tensor: (seq_len, num_features) """
    with tf.name_scope('apply_feature_mask'):
        shape = tf.shape(feature_tensor)
        full_mask = uniform_random_feature_mask(
            shape[0], shape[1], min_mask_len=min_mask_len, max_mask_len=max_mask_len)
        masked_tensor = tf.where(full_mask, feature_tensor, tf.zeros_like(feature_tensor, dtype=feature_tensor.dtype))
    return masked_tensor, full_mask


def build_dataset(data_files,
                  raw_data_spec,
                  shift,
                  input_feature_names,
                  target_feature_names,
                  seq_len,
                  min_seq_mask_len,
                  max_seq_mask_len,
                  min_feature_mask_len,
                  max_feature_mask_len,
                  feature_masked_ratio=0.2,
                  shuffle_buffer_size=None,
                  shuffle_files=False):
    feature_names = sorted(list(set(input_feature_names) | set(target_feature_names)))
    seq_columns_spec = sequence.SeqColumnsSpec(feature_names, seq_len, new_names='full', group=True)
    dataset = sequence.sequence_dataset(
        seq_columns_spec, data_files, raw_data_spec,
        shuffle_files=shuffle_files, shift=shift)

    def _add_mask(feature_dict):
        full_tensor = feature_dict['full']
        masked_tensor, full_mask = tf.cond(
            tf.random.uniform(()) > feature_masked_ratio,
            true_fn=lambda: _apply_sequence_mask(full_tensor, min_seq_mask_len, max_seq_mask_len),
            false_fn=lambda: _apply_feature_mask(full_tensor, min_feature_mask_len, max_feature_mask_len))
        return {'full': full_tensor, 'masked': masked_tensor, 'mask': full_mask}

    dataset = dataset.map(_add_mask)
    if shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size)

    return dataset


# a = tf.constant(1., )
# print(uniform_random_feature_mask(10, 5, 3, 4, tf.int32).numpy())

# feature_tensor = tf.constant(1., shape=(10, 10))
# tensor, mask = _apply_feature_mask(feature_tensor, 3, 5)
# print(tensor.numpy(), mask.numpy())

raw_data_spec = sequence.RawDataSpec(['c0', 'c1', 'c2'], [0., 0., 0.], 1, 30, False)
data_files = ['1.csv']
dataset = build_dataset(
    data_files,
    raw_data_spec,
    2,
    ['c0', 'c1'],
    ['c2'],
    4,
    2,
    3,
    1,
    2,
    feature_masked_ratio=0.5,
    shuffle_buffer_size=None,
    shuffle_files=False)
for x in dataset:
    tf.print('full:', x['full'].numpy())
    tf.print('masked:', x['masked'].numpy())
    tf.print('mask:', x['mask'].numpy())
    tf.print()
