""" Dataset utilities. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf


def named_dataset(dataset, names):
    """ Create a `Dataset` by adding feature names to an existed `Dataset`.
        Arguments:
            dataset: A `tuple` structured `Dataset`.
            names: A sequence of `str`, feature names.
        Returns:
            A `Dataset`.
    """
    return dataset.map(lambda *features: dict(zip(names, features)))


def feature_selected_dataset(dataset, selected_feature_names, output_is_tuple=False):
    """ Create a `Dataset` by selecting features from an existed `Dataset`.
        Arguments:
            dataset: A `dict` structured `Dataset`.
            selected_feature_names: A sequence of `str`, selected feature names.
            output_is_tuple: A `bool`, if true, return a `tuple` structured `Dataset`,
                or else, a `dict` structured one.
        Returns:
            A `Dataset`.
    """
    def map_fn(features):
        if output_is_tuple:
            return tuple(map(
                lambda k: features[k],
                selected_feature_names))
        return dict(map(
            lambda k: (k, features[k]),
            selected_feature_names))

    return dataset.map(map_fn)


def windowed_dataset(dataset,
                     size, shift=None, stride=1,
                     block_size=None, drop_remained_block=True,
                     name=None):
    """ Create a `Dataset` by windowing a sequential dataset.
        Arguments:
            dataset: A `Dataset` of output shape ((...), (...), ... (...)) or a `dict`
                of the same.
            size: A `tf.int64` scalar `tf.Tensor`, representing the number of elements
                of the input dataset to combine into a window.
            shift: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the forward
                shift of the sliding window in each iteration. Defaults to  `size`.
            stride: A `tf.int64` scalar `tf.Tensor`, representing the stride of the
                input elements in the sliding window.
            block_size: A `tf.int64` scalar `Tensor` like, block size. Windowing will be
                clipped among blocks.
            drop_remained_block: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
                whether a window should be dropped in case its size is smaller than
                `window_size`.
            name: A `str`, OP name, defaults to "windowed_dataset".
        Returns:
            A windowed `Dataset`.
    """
    def map_fn(*feature_tensors_or_dataset):
        if block_size is None:
            sub_dataset = feature_tensors_or_dataset[0]
        else:
            sub_dataset = tf.data.Dataset.from_tensor_slices(feature_tensors_or_dataset)
        sub_dataset = sub_dataset.window(size, shift=shift, stride=stride, drop_remainder=True)
        sub_dataset = sub_dataset.flat_map(
            lambda *tensors: tf.data.Dataset.zip(tuple(t.batch(size) for t in tensors)))
        return sub_dataset

    with tf.name_scope(name or 'windowed_dataset'):
        sorted_names = None
        output_types = tf.compat.v1.data.get_output_types(dataset)
        if type(output_types) is dict:
            sorted_names = sorted(output_types.keys())
            dataset = feature_selected_dataset(dataset, sorted_names, output_is_tuple=True)
        if block_size is None:
            tuple_dataset = map_fn(dataset)
        else:
            tuple_dataset = dataset.batch(block_size, drop_remainder=drop_remained_block).flat_map(map_fn)

        # Since `map_fn` deems inputs as a tuple, the output type should be recovered if necessary.
        dataset = tuple_dataset if type(tf.compat.v1.data.get_output_types(dataset)) is tuple \
            else tuple_dataset.map(lambda *f: f[0])
        if sorted_names is not None:
            dataset = named_dataset(dataset, sorted_names)
    return dataset
