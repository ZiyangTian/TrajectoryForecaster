import collections
import os
import tensorflow as tf

from forecaster.apps import base
from forecaster.apps.pretrain import masker as _masker
from forecaster.data import sequencer
from forecaster.models import layers
from forecaster.models import losses
from forecaster.models import networks
from forecaster.models import optimizers


def make_dataset(pattern,
                 raw_data_spec,
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
    column_spec = datasets.SequenceColumnsSpec(
        feature_names, sequence_length, group=True, new_names='features')

    with tf.name_scope(name or 'make_dataset'):
        dataset = datasets.sequence_dataset(
            [column_spec],
            data_files, raw_data_spec,
            shift=shift, stride=stride, shuffle_files=True,
            name='sequence_dataset')

        def add_mask(feature_dict):
            mask = masker.generate_mask((sequence_length, len(feature_names)))
            full_tensor = feature_dict['features']
            targets = tf.stack([full_tensor, tf.cast(mask, tf.float32)], axis=0)
            return (full_tensor, mask), targets

        dataset = dataset.map(add_mask, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if repeat_infinitely:
            dataset = dataset.repeat()
        if shuffle_buffer_size is not None:
            dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(batch_size)

    return dataset


class PreTrainer(tf.keras.Model):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_attention_heads,
                 conv_kernel_size,
                 num_features,
                 numeric_normalizer_fn=None,
                 numeric_restorer_fn=None,
                 name=None):
        super(PreTrainer, self).__init__(name=name or 'pre_trainer')
        self._encoder_layer = networks.SequenceEncoder(
            num_layers=num_layers, d_model=d_model,
            num_attention_heads=num_attention_heads, conv_kernel_size=conv_kernel_size,
            numeric_normalizer_fn=numeric_normalizer_fn, name='encoder')
        self._dense_layer = tf.keras.layers.Dense(num_features, name='dense')
        self._restorer_layer = layers.FunctionWrapper(numeric_restorer_fn, name='restorer')

    def call(self, inputs, **kwargs):
        encoded = self._encoder_layer(inputs)
        outputs = self._dense_layer(encoded)
        outputs = self._restorer_layer(outputs)
        return outputs


@base.app('pre_training')
class PreTraining(base.App):
    def build(self):
        data_config = self._config.data
        model_config = self._config.model
        normalizer_mean = tf.convert_to_tensor(list(map(
            lambda k: self._config.raw_data.features.__getattr__(k).mean, self._config.data.features)),
            dtype=tf.float32)
        normalizer_std = tf.convert_to_tensor(list(map(
            lambda k: self._config.raw_data.features.__getattr__(k).std, self._config.data.features)),
            dtype=tf.float32)

        def numeric_normalizer_fn(features):
            with tf.name_scope('numeric_normalizer_fn'):
                ans = (features - normalizer_mean) / normalizer_std
            return ans

        def numeric_restorer_fn(features):
            with tf.name_scope('numeric_restorer_fn'):
                ans = features * normalizer_std + normalizer_mean
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
            loss=losses.NormalizedMeanSquareErrorAtMask(numeric_normalizer_fn=numeric_normalizer_fn),
            metrics=[])
        return model

    @property
    def train_dataset(self):
        data_config = self._config.data
        mask_config = data_config.mask
        train_config = self._config.run.train
        return make_dataset(
            os.path.join(self._config.raw_data.train_dir, '*.txt'),
            datasets.RawDataSpec.from_config(self._config.raw_data),
            data_config.features,
            data_config.sequence_length,
            _masker.Masker.from_config(mask_config),
            train_config.data_shift,
            data_config.stride,
            train_config.batch_size,
            repeat_infinitely=True,
            shuffle_buffer_size=train_config.shuffle_buffer_size,
            name='train_dataset')

    @property
    def valid_dataset(self):
        data_config = self._config.data
        mask_config = data_config.mask
        eval_config = self._config.run.eval
        return make_dataset(
            os.path.join(self._config.raw_data.eval_dir, '*.txt'),
            datasets.RawDataSpec.from_config(self._config.raw_data),
            data_config.features,
            data_config.sequence_length,
            _masker.Masker.from_config(mask_config),
            eval_config.data_shift,
            data_config.stride,
            eval_config.batch_size,
            repeat_infinitely=True,
            shuffle_buffer_size=None,
            name='valid_dataset')

    @property
    def test_dataset(self):
        data_config = self._config.data
        mask_config = data_config.mask
        predict_config = self._config.run.predict
        return make_dataset(
            os.path.join(self._config.raw_data.test_dir, '*.txt'),
            datasets.RawDataSpec.from_config(self._config.raw_data),
            data_config.features,
            data_config.sequence_length,
            _masker.Masker.from_config(mask_config),
            predict_config.data_shift,
            data_config.stride,
            predict_config.batch_size,
            repeat_infinitely=False,
            shuffle_buffer_size=None,
            name='test_dataset')
