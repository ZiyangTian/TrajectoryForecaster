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
    def train_input_fn(self):
        """Function which takes no argument and return a `Dataset` for training, which can
            be used in an `Estimator`. See `input_fn` in `tf.estimator.Estimator.train`.
        """
        raise NotImplementedError('BaseJob.train_input_fn')

    @property
    @abc.abstractmethod
    def eval_input_fn(self):
        """Function which takes no argument and return a `Dataset` for evaluation, which can
            be used in an `Estimator`. See `input_fn` in `tf.estimator.Estimator.eval`.
        """
        raise NotImplementedError('BaseJob.eval_input_fn')

    @property
    def config(self):
        return self._config

    @property
    def model(self):
        return self._model

    def fit(self):
        run_config = self._config.run
        self._model.fit(
            self.train_input_fn(),
            epochs=run_config.train.,
            verbose=2,
            callbacks=None,
            validation_split=0.0,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False)
        self.ready()
        train_config = self._config.run.train
        eval_config = self._config.run.eval
        train_spec = tf.estimator.TrainSpec(
            self.train_input_fn,
            max_steps=train_config.steps, hooks=None)
        eval_spec = tf.estimator.EvalSpec(
            self.eval_input_fn,
            steps=eval_config.steps,
            name=None,
            hooks=None,
            exporters=None,
            start_delay_secs=eval_config.training.eval_start_delay_secs,
            throttle_secs=eval_config.training.eval_throttle_secs)
        return tf.estimator.train_and_evaluate(self._model, train_spec, eval_spec)

    def predict(self):
        tf.logging.info('Optional method `%s.predict` is not implemented. Nothing done.' % str(self.__class__))

    def _build(self):
        """
        "distribute": {
             "type": null,
             "tf_config": {
                 "cluster": {
                     "worker": [
                         "localhost:12345",
                         "localhost:23456"
                     ]
                 },
                 "task": {
                     "type": "worker",
                     "index": 0
                 }
             }
        }
        """
        distribute_config = self._model.run.distribute
        if distribute_config.type is None:
            return self.build()


    def _parse_tf_run_config(self):
        strategy = self._parse_distribute_config()
        summary_config = self._parse_summary_config()
        return tf.estimator.RunConfig(
            session_config=None, train_distribute=strategy, **summary_config)

    def _parse_distribute_config(self):
        distribute_config = self._config.run.get('distribute', None)
        if not distribute_config:
            return
        strategy = distributed.get_strategy(distribute_config.strategy)
        tf_config = {'cluster': distribute_config.cluster, 'task': distribute_config.task}
        os.environ['TF_CONFIG'] = json.dumps(tf_config)
        return strategy

    def _parse_summary_config(self):
        summary_config = self._config.run.get('summary', None).copy()
        if not summary_config:
            return
        save_checkpoints_every = summary_config.pop('save_checkpoints_every', 100)
        save_checkpoints_units = summary_config.pop('save_checkpoints_units', 'steps')
        if save_checkpoints_units == 'steps':
            summary_config['save_checkpoints_steps'] = save_checkpoints_every
        elif save_checkpoints_units in {'secs', 'seconds'}:
            summary_config['save_checkpoints_secs'] = save_checkpoints_every
        else:
            raise ValueError('Invalid value of key `save_checkpoints_units`.')
        return summary_config


class AbstractSequencesApp(AbstractApp, abc.ABC):
    def __init__(self, config):
        super(AbstractSequencesApp, self).__init__(config)
        data_config = self._config.data
        self._raw_data_spec = sequence.RawDataSpec(
            data_config.columns, data_config.column_defaults,
            data_config.stride, data_config.file_length, False)



