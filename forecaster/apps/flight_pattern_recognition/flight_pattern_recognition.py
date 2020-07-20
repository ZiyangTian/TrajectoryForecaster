import functools
import tensorflow as tf

from forecaster.apps import base
from forecaster.data import datasets
from forecaster.data import dataset_utils
from forecaster.models import layers
from forecaster.models import losses
from forecaster.models import metrics
from forecaster.models import networks
from forecaster.models import optimizers


def make_dataset(file_pattern_list,
                 column_names,
                 column_defaults,
                 feature_names,
                 sequence_length,
                 shift,
                 stride,
                 batch_size,
                 repeat=1,
                 sequence_shuffle_buffer_size=None,
                 example_shuffle_buffer_size=None,
                 name=None,
                 num_parallel_calls=None):

    for cls, file_pattern in enumerate(file_pattern_list):
        data_files = tf.io.gfile.glob(file_pattern)
        data_files_dataset = (tf.data.Dataset
                              .from_tensor_slices(data_files)
                              .map(lambda f: (f, cls), num_parallel_calls=num_parallel_calls))
        


    full_column_spec = datasets.SequenceColumnsSpec(
        list(feature_names) + list(other_feature_names),
        sequence_length + output_sequence_length,
        group=True, new_names='full_sequence')

    def generate_targets(feature_dict):
        inputs = feature_dict['full_sequence']
        targets = inputs[-output_sequence_length:, :len(feature_names)]
        return inputs, targets

    with tf.name_scope(name or 'make_dataset'):
        datasets = tf.data.Dataset.from_tensor_slices(data_files)  # 文件名 -> 数据集
        datasets = datasets.map(
            functools.partial(tf.data.experimental.CsvDataset, record_defaults=column_defaults, header=False),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)  # 载入文件内容
        datasets = datasets.map(
            functools.partial(dataset_utils.named_dataset, names=column_names),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)  # 加入特征名
        dataset = datasets.make_sequential_dataset(
            datasets,
            [full_column_spec],
            shift=shift,
            stride=stride,
            shuffle_buffer_size=len(data_files),
            cycle_length=len(data_files),
            block_length=1,
            name=None)
        dataset = dataset.map(generate_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if repeat_infinitely:
            dataset = dataset.repeat()
        if example_shuffle_buffer_size is not None:
            dataset = dataset.shuffle(example_shuffle_buffer_size)
        dataset = dataset.batch(batch_size)

    return dataset