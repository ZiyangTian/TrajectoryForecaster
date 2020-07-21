"""Sequencer for creating sequence datasets. """
import numpy as np
import tensorflow as tf

from forecaster.data import columns_specs as _columns_specs
from forecaster.data import dataset_utils
from utils import typing as typing_utils


class Sequencer(object):
    """Sequencer that map a set of datasets to a sequenced dataset.
        Arguments:
            columns_specs: A `list` of `ColumnsSpec`s.
            datasets: A `Dataset` of `dict` structured `Dataset`.
    """
    def __init__(self, columns_specs=None, datasets=None):
        self._columns_specs = typing_utils.normalize_list_of_type(
            columns_specs, _columns_specs.ColumnsSpec, allow_empty=True)
        self._check_repetitive_new_names()
        self._datasets = datasets

    def add(self, columns_spec):
        """Add a columns spec.
            Arguments:
                columns_spec: A `ColumnsSpec`.
            Returns:
                The instance itself.
        """
        self._columns_specs.append(columns_spec)
        self._check_repetitive_new_names()
        return self

    def from_files(self, files, load_fn, num_parallel_calls=None):
        """Update datasets by loading data from files.
            Arguments:
                files: A `list` of data file names, each of which represents an identical sequence.
                load_fn: A callable, mapping function to load data from a file.
                num_parallel_calls: See `tf.data.Dataset.map`.
            Returns:
                The instance itself.
        """
        self._datasets = tf.data.Dataset.from_tensor_slices(files).map(load_fn, num_parallel_calls=num_parallel_calls)
        return self

    def from_csv(self, files, column_names, num_parallel_calls=None, **kwargs):
        """Update datasets by loading CSV files.
            Arguments:
                files: A `list` of data file names, each of which represents an identical sequence.
                column_names: A `list` of `str`, column names of the CSV files.
                num_parallel_calls: See `tf.data.Dataset.map`.
                kwargs: Arguments in `tf.data.experimental.CSVDataset` other than `filenames`.
            Returns:
                The instance itself.
        """
        def load_fn(file):
            return dataset_utils.named_dataset(
                tf.data.experimental.CsvDataset([file], **kwargs),
                column_names,
                num_parallel_calls=num_parallel_calls)

        self.from_files(files, load_fn, num_parallel_calls=num_parallel_calls)
        return self

    def sequence_dataset(self,
                         shift=None,
                         stride=1,
                         cycle_length=1,
                         block_length=1,
                         num_parallel_calls=None):
        """Convert datasets to a single dataset with sequence features.
            Arguments:
                shift: An `int`, use sample every this number of raw samples, see
                    `tf.data.Dataset.generate`.
                stride: A `int`, the stride of the input elements in the sliding generate, see
                    `tf.data.Dataset.generate`.
                cycle_length: Argument for interleaving datasets. The number of input elements
                    that will be processed concurrently. If not set, the tf.data runtime decides
                    what it should be based on available CPU. If num_parallel_calls is set to
                    `tf.data.experimental.AUTOTUNE`, the cycle_length argument identifies the
                    maximum degree of parallelism.
                block_length: Argument for interleaving datasets. The number of consecutive
                    elements to produce from each input element before cycling to another input
                    element. If not set, defaults to 1.
                num_parallel_calls: Argument for interleaving datasets. If specified, the
                    implementation creates a threadpool, which is used to fetch inputs from
                    cycle elements asynchronously and in parallel. The default behavior is to
                    fetch inputs from cycle elements synchronously with no parallelism. If the
                    value `tf.data.experimental.AUTOTUNE` is used, then the number of parallel
                    calls is set dynamically based on available CPU.
            Returns:
                A `Dataset`.
        """
        if self._datasets is None:
            raise RuntimeError('Datasets have not been defined.')
        used_column_names = set(sum(map(lambda c: c.raw_columns, self._columns_specs), []))
        self._check_repetitive_new_names()
        max_offset = max(map(lambda c: c.max_offset, self._columns_specs))

        def feature_selected_map_fn(d):
            return dataset_utils.feature_selected_dataset(d, used_column_names, output_is_tuple=False)

        def convert_map_fn(windowed_dict):
            columns = {}
            for spec in self._columns_specs:
                columns.update(spec.convert(windowed_dict))
            return columns

        dataset = self._datasets.map(feature_selected_map_fn, num_parallel_calls=num_parallel_calls)
        dataset = dataset.interleave(
            lambda d: dataset_utils.windowed_dataset(
                d, max_offset, shift=shift, stride=stride, drop_remainder=True),
            cycle_length=cycle_length,
            block_length=block_length,
            num_parallel_calls=num_parallel_calls)
        dataset = dataset.map(convert_map_fn, num_parallel_calls=num_parallel_calls)
        return dataset

    def multi_sequence_dataset(self,
                               dataset_of_sequences,
                               columns_specs,
                               num_sequences,
                               sequence_id_name='seq_id',
                               shift=None,
                               stride=1,
                               shuffle_buffer_size=None,
                               repeat_sequences=1,
                               num_parallel_calls=None,
                               name=None):

        """
        Create multi-sequence dataset from CSV data files.
            Arguments:
                dataset_of_sequences: A finite `Dataset` of `dict` structured `Dataset`s, each of
                    which represents a sequence.
                columns_specs: A(n) (container of) `ColumnsSpec` instances.
                num_sequences: An `int`, number of datasets.
                sequence_id_name: A `str`, name of the sequence ID feature.
                shift: An `int`, use sample every this number of raw samples.
                stride: A `int`, the stride of the input elements in the sliding generate.
                shuffle_buffer_size: A `bool`, whether to shuffle data files.
                repeat_sequences: An `int`, the number of times the data files should be repeated. Defaults
                    to infinite times.
                num_parallel_calls:
                name: A `str`, OP name, defaults to "multi_sequence_dataset".
            Returns:
                A multi-sequence `Dataset`.
        """
        columns_specs = typing_utils.normalize_list_of_type(columns_specs, _columns_specs.ColumnsSpec)
        used_column_names = set(sum(map(lambda c: c.raw_columns, columns_specs), []))
        self._check_repetitive_new_names(columns_specs)  # , banned=sequence_id_name)
        max_offset = max(map(lambda c: c.max_offset, columns_specs))

        def feature_selected_map_fn(dataset):
            dataset = dataset_utils.feature_selected_dataset(dataset, used_column_names, output_is_tuple=False)
            return dataset

        def dataset_length_map_fn(dataset):
            return dataset.reduce(0, lambda count, _: count + 1)

        def process_map_fn(windowed_dict):
            columns = {}
            for spec in columns_specs:
                columns.update(spec.convert(windowed_dict))
            return columns

        def update_id_map_fn(id_tensor, features_dict):
            features_dict.update({sequence_id_name: id_tensor})
            return features_dict

        def process_batch_map_fn(id_batch, dataset_batch):
            datasets = dataset_batch.unbatch()
            min_length = datasets.map(dataset_length_map_fn).reduce(np.inf, min)

            sequential_dataset = datasets.interleave(
                lambda dataset: dataset_utils.windowed_dataset(
                    dataset, max_offset, shift=shift, stride=stride, drop_remainder=True),
                cycle_length=num_sequences, block_length=1, deterministic=True)
            sequential_dataset = sequential_dataset.map(process_map_fn, num_parallel_calls=num_parallel_calls)
            sequential_dataset = sequential_dataset.batch(num_sequences).take(min_length)

            ids = id_batch.repeat(min_length)
            dataset_with_id = tf.data.Dataset.zip((ids, sequential_dataset))
            dataset_with_id = dataset_with_id.map(update_id_map_fn, num_parallel_calls=num_parallel_calls)
            return dataset_with_id

        with tf.name_scope(name or 'multi_sequence_dataset'):
            dataset_of_sequences = dataset_of_sequences.map(
                feature_selected_map_fn, num_parallel_calls=num_parallel_calls)
            num_files = dataset_length_map_fn(dataset_of_sequences)
            id_dataset = tf.data.Dataset.range(num_files)
            dataset_of_sequences_with_ids = tf.data.Dataset.zip((dataset_of_sequences, id_dataset))
            dataset_of_sequences_with_ids = dataset_of_sequences_with_ids.repeat(repeat_sequences)
            if shuffle_buffer_size is not None:
                dataset_of_sequences_with_ids = dataset_of_sequences_with_ids.shuffle(num_files)

            dataset_of_sequences_with_ids = dataset_of_sequences_with_ids.batch(num_sequences, drop_remainder=True)
            dataset_of_sequences_with_ids = dataset_of_sequences_with_ids.flat_map(process_batch_map_fn)
        return dataset_of_sequences_with_ids

    def _check_repetitive_new_names(self, banned=None):
        banned = typing_utils.normalize_list_of_type(banned, str, allow_empty=True)
        for columns_spec in self._columns_specs:
            banned.extend(typing_utils.normalize_list_of_type(columns_spec.new_columns, str))
        typing_utils.normalize_list_of_type(banned, str, allow_empty=False, allow_duplicated=False)
