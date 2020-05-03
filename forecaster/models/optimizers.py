""" Optimizers. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf


_REGISTERED_OPTIMIZERS = {
    'adam': tf.compat.v1.train.AdamOptimizer,
}


def get_optimizer(config):
    config = config.copy()
    optimizer_type = config.pop('type')
    return _REGISTERED_OPTIMIZERS[optimizer_type](**config)
