import functools
import tensorflow as tf

from forecaster.apps import base
from forecaster.data import columns_specs
from forecaster.data import sequencer
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
                 cycle_length=1,
                 block_size=1,
                 repeat=None,
                 dataset_shuffle_buffer_size=None,
                 example_shuffle_buffer_size=None,
                 num_parallel_calls=None):
    files_list = list(map(lambda p: tf.io.gfile.glob(p), file_pattern_list))
    files = tf.constant(files_list, dtype=tf.string)
    modes = tf.range(tf.shape(files)[0])
    mode_dataset = tf.data.Dataset.from_tensor_slices(modes).map(lambda m: (m, files[m]))

    def mode_map_fn(m, fs):
        data_files_dataset = tf.data.Dataset.from_tensor_slices(fs)
        if dataset_shuffle_buffer_size is not None:
            data_files_dataset = data_files_dataset.shuffle(dataset_shuffle_buffer_size)

        def add_class_map_fn(feature_dict):
            feature_dict.update(mode=m)
            return feature_dict

        def load_fn(file):
            d = tf.data.experimental.CsvDataset([file], record_defaults=column_defaults)
            d = dataset_utils.named_dataset(d, column_names)
            d = d.map(add_class_map_fn, num_parallel_calls=num_parallel_calls)
            return d

        return data_files_dataset.map(load_fn, num_parallel_calls=num_parallel_calls)  # Dataset of datasets

    datasets = mode_dataset.interleave(
        mode_map_fn, cycle_length=len(file_pattern_list),
        block_length=1, num_parallel_calls=num_parallel_calls)
    full_column_spec = columns_specs.SequentialColumnsSpec(
        feature_names, sequence_length, group=True, new_columns='full_sequence')
    label_column_spec = columns_specs.ReservingColumnSpec(
        ['mode'], group=False, new_columns=['mode'])

    dataset = (sequencer.Sequencer([full_column_spec, label_column_spec], datasets)
               .sequence_dataset(shift=shift, stride=stride, cycle_length=cycle_length,
                                 block_length=block_size, num_parallel_calls=num_parallel_calls)
               .map(lambda feature_dict: (feature_dict['full_sequence'], feature_dict['mode']))
               .repeat(repeat))
    if example_shuffle_buffer_size is not None:
        dataset = dataset.shuffle(example_shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    return dataset


class TrajectoryPatternRecognitionModel(tf.keras.Model):
    def __init__(self,
                 num_patterns,
                 num_layers,
                 d_model,
                 num_attention_heads,
                 conv_kernel_size,
                 input_shape=None):
        inputs = tf.keras.Input(shape=input_shape)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.sequence_encoder = networks.SequenceEncoder(
            num_layers, d_model, num_attention_heads, conv_kernel_size, name='sequence_encoder')
        self.dense = tf.keras.layers.Dense(num_patterns, name='dense')

        outputs = self.layers(inputs)
        outputs = self.sequence_encoder(outputs)
        outputs = self.dense(outputs[:, -1])
        super(TrajectoryPatternRecognitionModel, self).__init__(inputs=inputs, outputs=outputs)


@base.app('trajectory_pattern_recognition')
class TrajectoryPatternRecognition(base.App):
    def build(self):
        pass

    def train_dataset(self):
        pass

