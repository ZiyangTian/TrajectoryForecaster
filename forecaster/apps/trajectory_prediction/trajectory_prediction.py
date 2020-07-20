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


def make_dataset(file_pattern,
                 column_names,
                 column_defaults,
                 trajectory_feature_names,
                 other_feature_names,
                 input_sequence_length,
                 output_sequence_length,
                 shift,
                 stride,
                 batch_size,
                 repeat_infinitely=False,
                 shuffle_buffer_size=None,
                 name=None):
    data_files = tf.io.gfile.glob(file_pattern)
    full_column_spec = datasets.SequenceColumnsSpec(
        list(trajectory_feature_names) + list(other_feature_names),
        input_sequence_length + output_sequence_length,
        group=True, new_names='full_sequence')

    def generate_targets(feature_dict):
        inputs = feature_dict['full_sequence']
        targets = inputs[-output_sequence_length:, :len(trajectory_feature_names)]
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
        if shuffle_buffer_size is not None:
            dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.batch(batch_size)

    return dataset


class TrajectoryPredictor(tf.keras.Model):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_attention_heads,
                 conv_kernel_size,
                 output_sequence_length,
                 num_outputs,
                 mask,
                 numeric_normalizer_fn=None,
                 numeric_restorer_fn=None):
        super(TrajectoryPredictor, self).__init__()
        self._mask = mask
        self._encoder = networks.SequenceEncoder(
            num_layers=num_layers, d_model=d_model,
            num_attention_heads=num_attention_heads, conv_kernel_size=conv_kernel_size,
            numeric_normalizer_fn=numeric_normalizer_fn, name=None)
        self._head_dense = tf.keras.layers.Dense(num_outputs)
        self._output_sequence_length = output_sequence_length
        self._numeric_restorer = layers.FunctionWrapper(
            tf.identity if numeric_restorer_fn is None else numeric_restorer_fn,
            name='numeric_restorer')

    def call(self, inputs, **kwargs):
        encoded = self._encoder(inputs, mask=self._mask)
        with tf.name_scope('head'):
            outputs = self._head_dense(encoded)[:, -self._output_sequence_length:, :]
            outputs = self._numeric_restorer(outputs)
        return outputs


@base.app('trajectory_prediction')
class TrajectoryPrediction(base.App):
    def build(self):
        data_config = self._config.data
        model_config = self._config.model
        normalizer_mean = list(map(
            lambda k: self._config.raw_data.features.__getattr__(k).mean,
            data_config.trajectory_features + data_config.other_features))
        normalizer_std = list(map(
            lambda k: self._config.raw_data.features.__getattr__(k).std,
            data_config.trajectory_features + data_config.other_features))
        num_trajectory_features = len(data_config.trajectory_features)
        num_other_features = len(data_config.other_features)
        num_features = num_trajectory_features + num_other_features

        input_mask = tf.concat([
            tf.ones((data_config.input_sequence_length, num_features), dtype=tf.int32),
            tf.zeros((data_config.output_sequence_length, num_features), dtype=tf.int32)],
            axis=0)

        with tf.name_scope('get_normalizer_params'):
            input_normalizer_mean = tf.convert_to_tensor(normalizer_mean, dtype=tf.float32)
            input_normalizer_std = tf.convert_to_tensor(normalizer_std, dtype=tf.float32)
            output_normalizer_mean = input_normalizer_mean[:num_trajectory_features]
            output_normalizer_std = input_normalizer_std[:num_trajectory_features]

        model = TrajectoryPredictor(
            model_config.num_layers, model_config.d_model,
            model_config.num_attention_heads,
            model_config.conv_kernel_size,
            data_config.output_sequence_length,
            num_trajectory_features,
            input_mask,
            numeric_normalizer_fn=lambda x: (x - input_normalizer_mean) / input_normalizer_std,
            numeric_restorer_fn=lambda x: x * output_normalizer_std + output_normalizer_mean)

        model.compile(
            optimizers.get_optimizer(model_config.optimizer),
            loss=losses.NormalizedMeanSquareError(lambda x: (x - output_normalizer_mean) / output_normalizer_std),
            metrics=[
                metrics.DestinationDeviation(),
                metrics.MinDeviation(),
                metrics.MeanDeviation(),
                metrics.MaxDeviation()])
        return model

    @property
    def train_dataset(self):
        raw_data_config = self._config.raw_data
        data_config = self._config.data
        train_config = self._config.run.train
        return make_dataset(
            self._config.raw_data.train_pattern,
            raw_data_config.columns,
            list(map(lambda x: raw_data_config.features.__getattr__(x).default, raw_data_config.columns)),
            data_config.trajectory_features,
            data_config.other_features,
            data_config.input_sequence_length,
            data_config.output_sequence_length,
            train_config.data_shift,
            data_config.stride,
            train_config.batch_size,
            repeat_infinitely=True,
            shuffle_buffer_size=train_config.shuffle_buffer_size,
            name='train_dataset')

    @property
    def valid_dataset(self):
        raw_data_config = self._config.raw_data
        data_config = self._config.data
        eval_config = self._config.run.eval
        return make_dataset(
            self._config.raw_data.eval_pattern,
            raw_data_config.columns,
            list(map(lambda x: raw_data_config.features.__getattr__(x).default, raw_data_config.columns)),
            data_config.trajectory_features,
            data_config.other_features,
            data_config.input_sequence_length,
            data_config.output_sequence_length,
            eval_config.data_shift,
            data_config.stride,
            eval_config.batch_size,
            repeat_infinitely=True,
            shuffle_buffer_size=None,
            name='valid_dataset')

    @property
    def test_dataset(self):
        raw_data_config = self._config.raw_data
        data_config = self._config.data
        predict_config = self._config.run.predict
        return make_dataset(
            self._config.raw_data.test_pattern,
            raw_data_config.columns,
            list(map(lambda x: raw_data_config.features.__getattr__(x).default, raw_data_config.columns)),
            data_config.trajectory_features,
            data_config.other_features,
            data_config.input_sequence_length,
            data_config.output_sequence_length,
            predict_config.data_shift,
            data_config.stride,
            predict_config.batch_size,
            repeat_infinitely=False,
            shuffle_buffer_size=None,
            name='test_dataset')
