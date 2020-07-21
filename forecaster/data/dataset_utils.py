""" Dataset utilities. """
import functools
import tensorflow as tf


def named_dataset(dataset, names, num_parallel_calls=None):
    """Create a `Dataset` by adding nested feature names to an existed `Dataset`.
        Arguments:
            dataset: A nested `tuple` structured `Dataset`.
            names: A nested `tuple` of `str`, feature names.
            num_parallel_calls: See `tf.data.Dataset.map`.
        Returns:
            A `dict` structured `Dataset`.
    """
    def map_fn(_names, *features):
        features_dict = {}
        for name, feature in zip(_names, features):
            if type(name) is str:
                features_dict.update({name: feature})
            elif type(name) is tuple and type(feature) is tuple:
                features_dict.update(map_fn(name, *feature))
            else:
                raise ValueError('Unmatched feature names and values: {} {}.'.format(name, feature))
        return features_dict

    return dataset.map(functools.partial(map_fn, names), num_parallel_calls=num_parallel_calls)


def feature_selected_dataset(dataset, selected_feature_names, output_is_tuple=False, num_parallel_calls=None):
    """Create a `Dataset` by selecting features from an existed `Dataset`.
        Arguments:
            dataset: A `dict` structured `Dataset`.
            selected_feature_names: A sequence of `str`, selected feature names.
            output_is_tuple: A `bool`, if true, return a `tuple` structured `Dataset`,
                or else, a `dict` structured one.
            num_parallel_calls: See `tf.data.Dataset.map`.
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

    return dataset.map(map_fn, num_parallel_calls=num_parallel_calls)


def windowed_dataset(dataset, size, shift=None, stride=1, drop_remainder=True):
    """Create a windowed `Dataset`.
        Arguments:
            dataset: A `Dataset` of output shape ((...), (...), ... (...)) or a `dict`
                of the same.
            size: A `tf.int64` scalar `tf.Tensor`, representing the number of elements
                of the input dataset to combine into a generate.
            shift: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the forward
                shift of the sliding generate in each iteration. Defaults to  `size`.
            stride: A `tf.int64` scalar `tf.Tensor`, representing the stride of the
                input elements in the sliding generate.
            drop_remainder:
        Returns:
            A windowed `Dataset`.
    """
    dataset = dataset.window(size, shift=shift, stride=stride, drop_remainder=drop_remainder)

    def map_fn(nested_structure_of_datasets):
        """nested_structure_of_datasets -> dataset"""
        structure_type = type(nested_structure_of_datasets)
        if structure_type is dict:
            for k, v in nested_structure_of_datasets.items():
                nested_structure_of_datasets[k] = map_fn(v)
            return tf.data.Dataset.zip(nested_structure_of_datasets)
        if structure_type is tuple:
            return tf.data.Dataset.zip(tuple(map(map_fn, nested_structure_of_datasets)))
        return nested_structure_of_datasets.batch(size)

    if type(dataset.element_spec) is tuple:
        return dataset.flat_map(lambda *e: map_fn(e))
    return dataset.flat_map(map_fn)
