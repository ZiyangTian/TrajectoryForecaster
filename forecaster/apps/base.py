""" Base for applications. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import json
import os
import attrdict
import tensorflow as tf

from forecaster import apps
from forecaster.data import sequence
from forecaster.run import distributed

REGISTERED_APPS = {}


class AppRegister(object):
    def __init__(self, name, short_name=None):
        self._name = name
        self._short_name = short_name

    def __call__(self, cls):
        setattr(apps, self._name, cls)
        if self._short_name is not None:
            REGISTERED_APPS.update({self._short_name: cls})
        return cls


app_register = AppRegister


class AbstractApp(object):
    """Base class for applications.
    Arguments:
        config: A `dict` like, configuration to build the job.
    """
    _engine_schema = None  # Reserved for schema.

    def __init__(self, config):
        self._config = attrdict.AttrDict(config)
        self._model = None
        self._build()

    @abc.abstractmethod
    def build(self):
        """Build `_model` and compile, do not include distribute strategy.
        Code example:
            self._model = tf.keras.Model(...)
            self._model.compile(...)
        """
        raise NotImplementedError('AbstractApp.build')

    @property
    @abc.abstractmethod
    def train_dataset(self):
        """Function which takes no argument and return a `Dataset` for training, which can
            be used in a `tf.keras.Model`. See `tf.keras.Model.fit`.
        """
        raise NotImplementedError('BaseJob.train_input_fn')

    @property
    @abc.abstractmethod
    def eval_dataset(self):
        """Function which takes no argument and return a `Dataset` for evaluation, which can
            be used in an `tf.keras.Model`. See `tf.keras.Model.fit`.
        """
        raise NotImplementedError('BaseJob.eval_input_fn')

    @property
    def predict_dataset(self):
        tf.logging.error('Property `predict_dataset` is not implemented, nothing returned.')
        return None

    @property
    def config(self):
        return self._config

    @property
    def model(self):
        return self._model

    def fit(self):
        run_config = self._config.run
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(run_config.model_path, 'ckpt_{epoch}'),
                save_best_only=True),
            tf.keras.callbacks.TensorBoard(
               log_dir=run_config.model_path,
               histogram_freq=1,
               write_graph=True,
               write_images=True,
               update_freq=run_config.save_summary_steps)]
        return self._model.fit(
            self.train_dataset,
            epochs=run_config.train.num_epochs,
            verbose=1,
            callbacks=callbacks,
            validation_data=self.eval_dataset,
            shuffle=True,
            steps_per_epoch=run_config.train.epoch_size,
            validation_steps=run_config.eval.steps,
            validation_freq=run_config.eval.every_epochs)

    def predict(self):
        predict_config = self._config.run.predict
        self._model.predict(
            self.predict_dataset,
            batch_size=predict_config.batch_size,
            steps=predict_config.steps)

    def _build(self):
        distribute_config = self._model.run.distribute
        distribute_type = distribute_config.type

        if distribute_type is None:
            return self.build()

        communication = getattr(tf.distribute.experimental.CollectiveCommunication, distribute_type.upper())
        tf_config = distribute_config.tf_config
        os.environ['TF_CONFIG'] = json.dumps(tf_config)
        strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(communication=communication)
        with strategy.scope():
            self.build()

    def _parse_distribute_config(self):
        distribute_config = self._config.run.get('distribute', None)
        if not distribute_config:
            return
        strategy = distributed.get_strategy(distribute_config.strategy)
        tf_config = {'cluster': distribute_config.cluster, 'task': distribute_config.task}
        os.environ['TF_CONFIG'] = json.dumps(tf_config)
        return strategy


class AbstractSequencesApp(AbstractApp, abc.ABC):
    def __init__(self, config):
        super(AbstractSequencesApp, self).__init__(config)
        data_config = self._config.data
        self._raw_data_spec = sequence.RawDataSpec(
            data_config.columns, data_config.column_defaults,
            data_config.stride, data_config.file_length, False)



