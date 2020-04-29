""" Base for applications. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import json
import os
import functools
import attrdict
import tensorflow as tf

from forecaster import apps
from forecaster.data import sequence

REGISTERED_MODELS = {}
REGISTERED_APPS = {}


class _Register(object):
    def __init__(self, container, name, short_name=None):
        self._name = name
        self._container = container
        self._short_name = short_name

    def __call__(self, cls):
        if hasattr(apps, self._name):
            raise AttributeError('{} has already existed.'.format(self._name))
        setattr(apps, self._name, cls)
        if self._short_name is not None:
            if self._container.get(self._short_name) is not None:
                raise ValueError('{} has already existed.'.format(self._short_name))
            self._container.update({self._short_name: cls})
        return cls


model = functools.partial(_Register, REGISTERED_MODELS)
app = functools.partial(_Register, REGISTERED_APPS)


class AbstractApp(object):
    """Base class for applications.
    Arguments:
        config: A `dict` like, configuration to build the job.
        warm_start_from: A `str`, representing the saved models file to warm-start from.
    """
    _engine_schema = None  # Reserved for schema.

    def __init__(self, config, warm_start_from=None):
        self._config = attrdict.AttrDict(config)
        self._distribute_strategy = None
        self._model = None
        self._build(warm_start_from=warm_start_from)

    @abc.abstractmethod
    def build(self):
        """Build `_model` and compile, do not include distribute strategy here.
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
        """Function which takes no argument and return a `Dataset` for predicting, which can
            be used in an `tf.keras.Model`. See `tf.keras.Model.predict`.
        """
        tf.logging.error('Property `predict_dataset` is not implemented, nothing returned.')
        return None

    @property
    def config(self):
        return self._config

    @property
    def model(self):
        return self._model

    def train(self):
        return self._fit(with_eval=False)

    def evaluate(self):
        return self._model.evaluate(
            self.eval_dataset, verbose=1, steps=self._config.run.eval.steps)

    def train_eval(self):
        return self._fit(with_eval=True)

    def predict(self):
        predict_config = self._config.run.predict
        self._model.predict(
            self.predict_dataset,
            batch_size=predict_config.batch_size,
            steps=predict_config.steps)

    def _build(self, warm_start_from=None):
        distribute_config = self._config.run.distribute
        distribute_type = distribute_config.type

        if distribute_type is None:
            self.build()
            if warm_start_from is not None:
                self._model.load_weights(warm_start_from)
            return

        communication = getattr(tf.distribute.experimental.CollectiveCommunication, distribute_type.upper())
        tf_config = distribute_config.tf_config
        os.environ['TF_CONFIG'] = json.dumps(dict(tf_config))
        self._distribute_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(communication=communication)
        with self._distribute_strategy.scope():
            self.build()
            if tf_config.task.index == 0 and warm_start_from is not None:
                self._model.load_weights(warm_start_from)

    def _fit(self, with_eval=True):
        run_config = self._config.run
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self._config.path.checkpoints_dir, 'ckpt_{epoch}'),
                save_best_only=True, save_freq=run_config.save.save_checkpoints_every),
            tf.keras.callbacks.TensorBoard(
               log_dir=self._config.path.checkpoints_dir,
               histogram_freq=1,
               write_graph=True,
               write_images=True,
               update_freq=run_config.save.save_summary_steps)]
        return self._model.fit(
            self.train_dataset,
            epochs=run_config.train.num_epochs,
            verbose=1,
            callbacks=callbacks,
            validation_data=self.eval_dataset if with_eval else None,
            shuffle=True,
            steps_per_epoch=run_config.train.epoch_size,
            validation_steps=run_config.eval.steps if with_eval else None,
            validation_freq=run_config.eval.every_epochs if with_eval else None)


class AbstractSequencesApp(AbstractApp, abc.ABC):
    def __init__(self, config):
        super(AbstractSequencesApp, self).__init__(config)

        columns = self._config.raw_data.columns
        column_defaults = list(map(lambda k: self._config.raw_data.features[k]['default'], columns))

        self._raw_data_spec = sequence.RawDataSpec(
            columns, column_defaults,
            self._config.data.stride, self._config.raw_data.file_length, False)



