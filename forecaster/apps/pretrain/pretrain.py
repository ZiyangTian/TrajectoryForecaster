import collections
import os
import tensorflow as tf

from forecaster.apps import base
from forecaster.apps.pretrain import masker as _masker
from forecaster.data import sequence
from forecaster.models import layers
from forecaster.models import networks
from forecaster.models import optimizers


def make_dataset(pattern,
                 raw_data_spec: sequence.RawDataSpec,
                 feature_names,
                 sequence_length,
                 masker: _masker.Masker,
                 shift,
                 stride,
                 batch_size,
                 repeat_infinitely=False,
                 shuffle_buffer_size=None,
                 name=None):
    data_files = tf.io.gfile.glob(pattern)
    column_spec = sequence.SeqColumnsSpec(
        feature_names, sequence_length, group=True, new_names='features')

    with tf.name_scope(name or 'make_dataset'):
        dataset = sequence.sequence_dataset(
            [column_spec],
            data_files, raw_data_spec,
            shift=shift, stride=stride, shuffle_files=True,
            name='sequence_dataset')
        dataset = dataset.map(
            lambda feature_dict: (
                (feature_dict['features'], masker.generate_mask((sequence_length, len(feature_names)))),
                feature_dict['features']),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.cache()
        if repeat_infinitely:
            dataset = dataset.repeat()
        if shuffle_buffer_size is not None:
            dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(batch_size)

    return dataset


class PreTrainer(tf.keras.Sequential):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_attention_heads,
                 conv_kernel_size,
                 num_features,
                 numeric_normalizer_fn=None,
                 numeric_restorer_fn=None):
        encoder_layer = networks.SequenceEncoder(
            num_layers=num_layers, d_model=d_model,
            num_attention_heads=num_attention_heads, conv_kernel_size=conv_kernel_size,
            numeric_normalizer_fn=numeric_normalizer_fn, name=None)
        dense_layer = tf.keras.layers.Dense(num_features, name='dense')
        restorer_layer = layers.FunctionWrapper(numeric_restorer_fn, name='restorer')
        super(PreTrainer, self).__init__(layers=[encoder_layer, dense_layer, restorer_layer], name='pre_trainer')


@base.app('pre_training')
class PreTraining(base.App):
    def build(self):
        data_config = self._config.data
        model_config = self._config.model
        normalizer_mean = list(map(
            lambda k: self._config.raw_data.features.__getattr__(k).mean, self._config.data.features))
        normalizer_std = list(map(
            lambda k: self._config.raw_data.features.__getattr__(k).std, self._config.data.features))

        def numeric_normalizer_fn(features):
            with tf.name_scope('numeric_normalizer_fn'):
                mean = tf.constant(normalizer_mean, dtype=tf.float32)
                std = tf.constant(normalizer_std, dtype=tf.float32)
                ans = (features - mean) / std
            return ans

        def numeric_restorer_fn(features):
            with tf.name_scope('numeric_restorer_fn'):
                mean = tf.convert_to_tensor(normalizer_mean, dtype=tf.float32)
                std = tf.convert_to_tensor(normalizer_std, dtype=tf.float32)
                ans = features * std + mean
            return ans

        model = PreTrainer(
            model_config.num_layers, model_config.d_model,
            model_config.num_attention_heads,
            model_config.conv_kernel_size,
            len(data_config.features),
            numeric_normalizer_fn=numeric_normalizer_fn,
            numeric_restorer_fn=numeric_restorer_fn)
        model.compile(
            optimizers.get_optimizer(model_config.optimizer),
            loss='categorical_crossentropy',
            metrics=['binary_accuracy'])
        return model

    @property
    def train_dataset(self):
        data_config = self._config.data
        train_config = self._config.run.train
        return make_dataset(
            os.path.join(self._config.raw_data.train_dir, '*.csv'),
            sequence.RawDataSpec.from_config(self._config.raw_data),
            data_config.features,
            data_config.labels,
            data_config.sequence_length,
            train_config.data_shift,
            data_config.stride,
            train_config.batch_size,
            repeat_infinitely=True,
            shuffle_buffer_size=train_config.shuffle_buffer_size,
            name='train_dataset')

    @property
    def valid_dataset(self):
        data_config = self._config.data
        eval_config = self._config.run.eval
        return make_dataset(
            os.path.join(self._config.raw_data.eval_dir, '*.csv'),
            sequence.RawDataSpec.from_config(self._config.raw_data),
            data_config.features,
            data_config.labels,
            data_config.sequence_length,
            eval_config.data_shift,
            data_config.stride,
            eval_config.batch_size,
            repeat_infinitely=False,
            shuffle_buffer_size=None,
            name='valid_dataset')

    @property
    def test_dataset(self):
        data_config = self._config.data
        predict_config = self._config.run.predict
        return make_dataset(
            os.path.join(self._config.raw_data.test_dir, '*.csv'),
            sequence.RawDataSpec.from_config(self._config.raw_data),
            data_config.features,
            data_config.labels,
            data_config.sequence_length,
            predict_config.data_shift,
            data_config.stride,
            predict_config.batch_size,
            repeat_infinitely=False,
            shuffle_buffer_size=None,
            name='test_dataset')
