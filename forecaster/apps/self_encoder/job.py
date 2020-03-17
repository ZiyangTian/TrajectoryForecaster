import copy
import os
import tensorflow as tf

from forecaster.apps import base
from forecaster.apps.self_encoder import data
from forecaster.apps.self_encoder import models
from forecaster.model import metrics
from forecaster.model import optimizers


@base.app_register('SelfEncoderJob')
class SelfEncoderJob(base.AbstractApp):
    def __init__(self, config, build=True):
        super(SelfEncoderJob, self).__init__(config, build=build)

    def build(self):
        model_config = self._config.model
        self._model = models.SequenceSelfEncoder(
            model_config.params.encoder_hidden_sizes,
            model_config.params.encoder_pooling_sizes,
            model_config.params.encoder_kernel_sizes,
            model_config.params.decoder_hidden_sizes,
            model_config.params.decoder_strides,
            model_config.params.decoder_kernel_sizes,
            model_config.params.target_size,
            uniform_centre=None,
            uniform_bound=None,
            restore_centre=None,
            restore_bound=None,
            name=None)
        self._model.compile(
            optimizer=optimizers.get_optimizer(dict(model_config.optimizer)),
            loss=[None, 'mse'],
            metrics=[
                [metrics.get_metrics(
                    'max_deviation',
                    'mean_deviation',
                    'min_deviation',
                    'destination_deviation')],
                None],
            loss_weights=None,
            sample_weight_mode=None,
            weighted_metrics=None)

    @property
    def train_dataset(self):
        raw_data_config = self._config.raw_data
        data_config = self._config.data
        return data.build_dataset(
            tf.io.gfile.glob(os.path.join(raw_data_config.train_dir, '*')),
            self._raw_data_spec,
            data_config.shift,
            data_config.input_features,
            data_config.target_features,
            data_config.seq_len,
            min_seq_mask_len=data_config.min_seq_mask_len,
            max_seq_mask_len=data_config.max_seq_mask_len,
            min_feature_mask_len=data_config.min_feature_mask_len,
            max_feature_mask_len=data_config.max_feature_mask_len,
            feature_masked_ratio=data_config.feature_masked_ratio,
            shuffle_buffer_size=10000,
            shuffle_files=True)

    @property
    def eval_dataset(self):
        raw_data_config = self._config.raw_data
        data_config = self._config.data
        return data.build_dataset(
            tf.io.gfile.glob(os.path.join(raw_data_config.eval_dir, '*')),
            self._raw_data_spec,
            data_config.shift,
            data_config.input_features,
            data_config.target_features,
            data_config.seq_len,
            min_seq_mask_len=data_config.min_seq_mask_len,
            max_seq_mask_len=data_config.max_seq_mask_len,
            min_feature_mask_len=data_config.min_feature_mask_len,
            max_feature_mask_len=data_config.max_feature_mask_len,
            feature_masked_ratio=data_config.feature_masked_ratio,
            shuffle_buffer_size=None,
            shuffle_files=False)
