import copy
import tensorflow as tf

from forecaster.apps.base import app_register
from forecaster.apps.self_encoder import data
from forecaster.apps.self_encoder import models


@app_register('SelfEncoderJob')
class SelfEncoderJob(object):
    def __init__(self, config):
        self._config = copy.copy(config)
        model_config = config.model
        self._model = models.SequenceSelfEncoder(
            model_config.encoder_hidden_sizes,
            model_config.encoder_pooling_sizes,
            model_config.encoder_kernel_sizes,
            model_config.decoder_hidden_sizes,
            model_config.decoder_strides,
            model_config.decoder_kernel_sizes,
            model_config.target_size,
            uniform_centre=None,
            uniform_bound=None,
            restore_centre=None,
            restore_bound=None,
            name=None)
        self._model.compile(
            optimizer='adam', loss=[None, 'mse'], metrics=None, loss_weights=None,
            sample_weight_mode=None, weighted_metrics=None, target_tensors=None,
            distribute=None, **kwargs
        )

        optimizer = tf.keras.optimizers.Optimizer.from_config(model_config.optimizer)
        losses = [None, 'mse']
        metrics =
