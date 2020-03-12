""" Optimizers. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import tensorflow as tf


_REGISTERED_OPTIMIZERS = {
    'adadelta': tf.keras.optimizers.Adadelta,
    'adagrad': tf.keras.optimizers.Adagrad,
    'adam': tf.keras.optimizers.Adam,
    'adamax': tf.keras.optimizers.Adamax,
    'ftrl': tf.keras.optimizers.Ftrl,
    'nadam': tf.keras.optimizers.Nadam,
    'rmsprop': tf.keras.optimizers.RMSprop,
    'sgd': tf.keras.optimizers.SGD
}


def get_optimizer(config):
    config = copy.deepcopy(config)
    optimizer_type = config.pop['type']
    return _REGISTERED_OPTIMIZERS[optimizer_type](**config)
