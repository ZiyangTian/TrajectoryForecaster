""" Base for applications. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import collections
import os
import json
import attrdict
import tensorflow as tf

from forecaster import apps

REGISTERED_APPS = {}


def get_app(name):
    return REGISTERED_APPS[name]


class AppRegister(collections.namedtuple('_Register', ('name',))):
    def __call__(self, cls):
        setattr(apps, self.name, cls)
        if REGISTERED_APPS.get(self.name) is not None:
            raise ValueError('{} has already existed.'.format(self.name))
        REGISTERED_APPS.update({self.name: cls})
        return cls


app = AppRegister


class App(object):
    """Base class for apps.
    Arguments:
        config: A `dict` like, configuration to build the job.
        warm_start_from: A `str`, representing the saved models file to warm-start from.
    """
    _engine_schema = None  # Reserved for schema.
    _schema = None

    def __init__(self, config, warm_start_from=None):
        self._config = attrdict.AttrDict(config)
        self._model = None
        self._build(warm_start_from=warm_start_from)

    @abc.abstractmethod
    def build(self):
        """Return a compiled `tf.keras.Model`, do not include distribute strategy here.
        Code example:
            model = tf.keras.Model(...)
            model.complile(...)
            return model
        """
        raise NotImplementedError('App.build')

    @property
    @abc.abstractmethod
    def train_dataset(self):
        """Returns a `Dataset` for training, which can
            be used in a `tf.keras.Model`. See `tf.keras.Model.fit`.
        """
        raise NotImplementedError('App.train_dataset')

    @property
    @abc.abstractmethod
    def valid_dataset(self):
        """Returns a `Dataset` for validation, which can
            be used in a `tf.keras.Model`. See `tf.keras.Model.fit`.
        """
        raise NotImplementedError('App.valid_dataset')

    @property
    @abc.abstractmethod
    def test_dataset(self):
        """Returns a `Dataset` for training, which can
            be used in a `tf.keras.Model`. See `tf.keras.Model.evaluate`.
        """
        return NotImplementedError('App.test_dataset')

    @property
    def config(self):
        return self._config

    @property
    def model(self):
        return self._model

    def fit(self):
        train_config = self._config.run.train
        path_config = self._config.path
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(path_config.checkpoints_dir, 'best.ckpt'),
                monitor='val_loss',
                verbose=0,
                save_best_only=True,
                save_freq='epoch',
                load_weights_on_restart=True),
            tf.keras.callbacks.TensorBoard(
                log_dir=path_config.tensorboard_dir,
                histogram_freq=1,
                update_freq='batch')]
        return self._model.fit(
            self.train_dataset, epochs=train_config.epochs, verbose=1, callbacks=callbacks,
            validation_data=self.valid_dataset,
            steps_per_epoch=train_config.steps_per_epoch)

    def evaluate(self):
        return self._model.evaluate(self.test_dataset, return_dict=True)

    def predict(self, *args, **kwargs):
        self._model.predict(*args, **kwargs)

    def _build(self, warm_start_from=None):
        del warm_start_from  # TODO: ...
        distribute_config = self._config.run.distribute
        distribute_type = distribute_config.type

        if distribute_type is not None:
            communication = getattr(tf.distribute.experimental.CollectiveCommunication, distribute_type.upper())
            tf_config = distribute_config.tf_config
            os.environ['TF_CONFIG'] = json.dumps(dict(tf_config))
            strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(communication)
            with strategy.scope():
                self._model = self.build()
        else:
            self._model = self.build()
