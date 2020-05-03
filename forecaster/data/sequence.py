""" Create sequence dataset from original CSV data files. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import collections
import attrdict
import tensorflow as tf

from forecaster.data import dataset_impl
from utils import typing as typing_utils


class ColumnsSpec(object):
    """ Column pre-processing spec base class. """
    def __init__(self, raw_column_names, group=True, new_names=None):
        self._raw_column_names = typing_utils.normalize_list_of_type(raw_column_names, str)
        self._group = group
        if new_names is None:
            new_names = self._generate_column_name()
        self._new_names = self._allocate_new_names(new_names)

    @property
    def new_names(self):
        return self._new_names

    @property
    @abc.abstractmethod
    def max_offset(self):
        raise NotImplementedError('ColumnsSpec.max_offset')

    @abc.abstractmethod
    def _new_name_prefix(self):
        raise NotImplementedError('ColumnsSpec._new_name_prefix')

    def _generate_column_name(self):
        if self._group:
            return '%s__%s' % (
                self._new_name_prefix(), '__'.join(self._raw_column_names))

    def _allocate_new_names(self, new_names):
        if self._group:
            if type(new_names) is not str:
                raise ValueError('`new_names` must be one `str` if grouping, got %s.' % type(new_names))
        else:
            new_names = typing_utils.normalize_list_of_type(new_names, str)
            if len(new_names) != len(self._raw_column_names):
                raise ValueError('Number of `new_names` elements must be equal to `raw_column_names`, '
                                 'got %i and %i.' % (len(new_names), len(self._raw_column_names)))
        return new_names

    @abc.abstractmethod
    def process_map_fn(self, windowed_dict, name):
        raise NotImplementedError('ColumnsSpec.process_columns')

    def get_new_names(self, original_name):
        if self._group:
            raise AttributeError('Connot `get_new_names` when `group` is true.')
        return self._new_names[self._raw_column_names.index(original_name)]


class SeqColumnsSpec(ColumnsSpec):
    """ Sequence columns spec.
        Arguments:
            raw_column_names: A `list` of `str`, raw column names.
            sequence_length: An `int`, sequence length.
            offset: An `int`, relative offset of sequence starting position.
            group: A `bool`, whether to group columns. If true, all raw columns must have
                identical data type.
            new_names: A (sequence of) `str`, new column name(s). If group is true, `new_names`
                must be a `str`; or else, its elements must match the order of `raw_column_names`.
                By default, automatically generate.
        Raises:
            ValueError: If number of `new_names` elements is not compatible with
                `raw_column_names` and `group`.
            AttributeError: If use method `get_new_name` when `group` is true.
    """
    def __init__(self, raw_column_names, sequence_length, offset=0, group=True, new_names=None):
        self._sequence_length = sequence_length
        self._offset = offset
        super(SeqColumnsSpec, self).__init__(raw_column_names, group=group, new_names=new_names)

    @property
    def max_offset(self):
        return self._sequence_length + self._offset

    def _new_name_prefix(self):
        return 'SeqColumns_%i_%i' % (self._sequence_length, self._offset)

    def process_map_fn(self, windowed_dict, name=None):
        """ Map from a windowed feature `dict` to sequential feature tensors.
            Arguments:
                windowed_dict: A `dict` from `str` feature names to windowed `Tensor`s.
                name: A `str`, OP name, defaults to "process_seq_columns".
            Returns:
                A `dict` from `str` feature names to sequential `Tensor`s.
        """
        with tf.name_scope(name or 'process_seq_columns'):
            processed_features = map(
                lambda k: windowed_dict[k][self._offset: self._offset + self._sequence_length],
                self._raw_column_names)
            if self._group:
                processed_features = {self._new_names: tf.stack(list(processed_features), axis=-1)}
            else:
                processed_features = dict(zip(self._new_names, processed_features))
        return processed_features


class ReducingColumnsSpec(ColumnsSpec):
    """ Sequence columns spec.
            Arguments:
                raw_column_names: A `list` of `str`, raw column names.
                use_avg: One or a pair of `int`(s).
                    `int`(n) -> reduce to the average of sequence [0: n];
                    `int`(n0, n1) -> reduce to the average of sequence [n1: n2];
                    defaults to reduce to the one in a specified position.
                rsv_pos: An `int`, relative offset of the position of the reserved value. If `use_avg`
                    is not `None`, neglect this argument.
                group: A `bool`, whether to group columns. If true, all raw columns must have
                    identical data type.
                new_names: A (sequence of) `str`, new column name(s). If group is true, `new_names`
                    must be a `str`; or else, its elements must match the order of `raw_column_names`.
                    By default, automatically generate.
            Raises:
                ValueError: If number of `new_names` elements is not compatible with
                    `raw_column_names` and `group`.
                AttributeError: If use method `get_new_name` when `group` is true.
    """
    def __init__(self, raw_column_names, use_avg=None, rsv_pos=0, group=True, new_names=None):
        self._use_avg = use_avg
        if use_avg is None:
            self._rsv_pos = rsv_pos
            self._avg_begin_pos, self._avg_end_pos = None, None
        else:
            self._rsv_pos = None
            if type(use_avg) is int:
                self._avg_begin_pos, self._avg_end_pos = 0, use_avg
            else:
                self._avg_begin_pos, self._avg_end_pos = tuple(use_avg)
        super(ReducingColumnsSpec, self).__init__(raw_column_names, group=group, new_names=new_names)

    @property
    def max_offset(self):
        return self._rsv_pos + 1 if self._use_avg is None else self._avg_end_pos + 1

    def _new_name_prefix(self):
        if self._use_avg is None:
            return 'RsvColumn_%s' % str(self._rsv_pos)
        return 'RsvColumn_%s_%s' % (str(self._avg_begin_pos), str(self._avg_end_pos))

    def process_map_fn(self, windowed_dict, name=None):
        """ Map from a windowed feature `dict` to sequential feature tensors.
            Arguments:
                windowed_dict: A `dict` from `str` feature names to windowed `Tensor`s.
                name: A `str`, OP name, defaults to "process_rsv_columns".
            Returns:
                A `dict` from `str` feature names to sequential `Tensor`s.
        """
        with tf.name_scope(name or 'precess_rsv_columns'):
            if self._use_avg is None:
                processed_features = map(lambda k: windowed_dict[k][self._rsv_pos], self._raw_column_names)
            else:
                processed_features = map(
                    lambda k: tf.reduce_mean(windowed_dict[k][self._avg_begin_pos: self._avg_end_pos]),
                    self._raw_column_names)
            if self._group:
                processed_features = {self._new_names: tf.stack(list(processed_features), axis=-1)}
            else:
                processed_features = dict(zip(self._new_names, processed_features))
        return processed_features


class RawDataSpec(collections.namedtuple(
    'RawDataSpec', (
        'column_names',
        'column_defaults',
        'block_size',
        'header'))):
    """ Spec for raw data files.
        Arguments:
            column_names: A sequence of `str`, column names that match the titles in the CSV files.
            column_defaults: A sequence , default column values that match `column_names`.
            block_size: An `int`, number of samples in a single data file.
            csv_with_header: A `bool`, whether the CSV files have column headers.
    """
    @classmethod
    def from_config(cls, config):
        config = attrdict.AttrDict(config)
        features_config = config.features
        column_defaults = list(map(lambda k: features_config.__getattr__(k).default, config.columns))
        return cls(config.columns, column_defaults, config.block_size, config.header)

    def dataset(self, data_files, shuffle=False, **kwargs):
        """ Create a dataset from data files.
            Arguments:
                data_files: A `str` or an 1-D `tf.string` `Tensor` like, data file names.
                shuffle: A `bool`, whether to shuffle the data files.
                kwargs: Keyword arguments, other configurations for creating `CsvDataset`.
            Returns:
                A `Dataset`.
        """
        if not tf.is_tensor(data_files):
            data_files = typing_utils.normalize_list_of_type(data_files, str)
        if shuffle:
            data_files = tf.random.shuffle(tf.convert_to_tensor(data_files, dtype=tf.string))
        return tf.data.experimental.CsvDataset(
            data_files,
            self.column_defaults,
            header=self.header,
            **kwargs)


def _exam_repetitive_new_names(columns_specs, banned=None):
    banned = typing_utils.normalize_list_of_type(banned, str, allow_empty=True)
    for columns_spec in columns_specs:
        banned.extend(typing_utils.normalize_list_of_type(columns_spec.new_names, str))
    typing_utils.normalize_list_of_type(banned, str, allow_empty=False, allow_duplicated=False)


def sequence_dataset(columns_specs,
                     data_files,
                     raw_data_spec: RawDataSpec,
                     shift=None,
                     stride=1,
                     shuffle_files=False,
                     name=None,
                     **kwargs):
    """ Create a sequence dataset from CSV data files.
        Arguments:
            columns_specs: A(n) (container of) `ColumnsSpec` instances.
            data_files: A `str` or an 1-D `tf.string` `Tensor` like, data file name(s).
            raw_data_spec: An instance of `RawDataSpec`.
            shift: An `int`, the forward shift of the sliding window.
            stride: A `int`, the stride of the input elements in the sliding window.
            shuffle_files: A `bool`, whether to shuffle data files.
            name: A `str`, OP name, defaults to "sequence_dataset".
            kwargs: Keyword arguments to create the `CsvDataset`.
        Returns:
            A sequence `Dataset`.
    """
    columns_specs = typing_utils.normalize_list_of_type(columns_specs, ColumnsSpec)
    _exam_repetitive_new_names(columns_specs)
    max_offset = max(map(lambda c: c.max_offset, columns_specs))

    def process_map_fn(windowed_dict):
        columns = {}
        for spec in columns_specs:
            columns.update(spec.process_map_fn(windowed_dict))
        return columns

    with tf.name_scope(name or 'sequence_dataset'):
        dataset = raw_data_spec.dataset(data_files, shuffle_files, **kwargs)
        dataset = dataset_impl.named_dataset(dataset, raw_data_spec.column_names)
        dataset = dataset_impl.windowed_dataset(
            dataset, max_offset, shift=shift, stride=stride,
            block_size=raw_data_spec.block_size, drop_remained_block=True)
        dataset = dataset.map(process_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


def multi_sequence_dataset(columns_specs,
                           data_files,
                           raw_data_spec: RawDataSpec,
                           num_seqs,
                           seq_id_name='seq_id',
                           shift=None,
                           stride=1,
                           shuffle_files=False,
                           repeat_files=1,
                           name=None,
                           **kwargs):
    """ Create multi-sequence dataset from CSV data files.
        Arguments:
            columns_specs: A(n) (container of) `ColumnsSpec` instances.
            data_files: A `str` or an 1-D `tf.string` `Tensor` like, data file name(s).
            raw_data_spec: An instance of `RawDataSpec`.
            num_seqs: An `int`, number of sequences.
            seq_id_name: A `str`, name of the sequence ID feature.
            shift: An `int`, use sample every this number of raw samples.
            stride: A `int`, the stride of the input elements in the sliding window.
            shuffle_files: A `bool`, whether to shuffle data files.
            repeat_files: An `int`, the number of times the data files should be repeated. Defaults
                to infinite times.
            name: A `str`, OP name, defaults to "multi_sequence_dataset".
            kwargs: Keyword arguments to create the `CsvDataset`.
        Returns:
            A multi-sequence `Dataset`.
    """
    columns_specs = typing_utils.normalize_list_of_type(columns_specs, ColumnsSpec)
    _exam_repetitive_new_names(columns_specs, banned=seq_id_name)
    data_files = typing_utils.normalize_list_of_type(data_files, str)
    max_offset = max(map(lambda c: c.max_offset, columns_specs))
    num_files = len(data_files)

    def load_data_map_fn(id_tensor):
        file_names_tensor = tf.gather(tf.convert_to_tensor(data_files), id_tensor)
        seqs_data = raw_data_spec.dataset(file_names_tensor, shuffle=shuffle_files, **kwargs)
        seqs_data = dataset_impl.windowed_dataset(seqs_data, raw_data_spec.block_size)
        seqs_data = dataset_impl.windowed_dataset(seqs_data, num_seqs)

        def map_fn(*bn_features):
            features_list = list(map(lambda fea: tf.transpose(fea), bn_features))
            features_list.append(tf.broadcast_to(file_names_tensor, (raw_data_spec.block_size, num_seqs)))
            return tuple(features_list)

        seqs_data = seqs_data.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        seqs_data = seqs_data.flat_map(lambda *features: tf.data.Dataset.from_tensor_slices(features))
        seqs_data = dataset_impl.named_dataset(seqs_data, list(raw_data_spec.column_names) + [seq_id_name])
        return seqs_data

    def process_map_fn(windowed_dict):
        columns = {}
        for spec in columns_specs:
            columns.update(spec.process_map_fn(windowed_dict))
        return columns

    with tf.name_scope(name or 'multi_sequence_dataset'):
        id_dataset = tf.data.Dataset.range(num_files)
        if shuffle_files:
            id_dataset = id_dataset.shuffle(num_files)
        id_dataset = id_dataset.repeat(repeat_files)
        id_dataset = dataset_impl.windowed_dataset(
            id_dataset,
            size=num_seqs, shift=shift, stride=1,
            block_size=num_files, drop_remained_block=True)
        dataset = id_dataset.flat_map(load_data_map_fn)
        dataset = dataset_impl.windowed_dataset(
            dataset, max_offset, shift=shift, stride=stride,
            block_size=raw_data_spec.block_size, drop_remained_block=True)
        dataset = dataset.map(process_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset
