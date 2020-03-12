""" Base for applications. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import copy
import json
import os
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


class BaseJob(object):
    def __init__(self, config, build=True):
        self._config = copy.copy(config)
        data_config = self._config.data
        self._raw_data_spec = sequence.RawDataSpec(
            data_config.columns, data_config.column_defaults,
            data_config.stride, data_config.file_length, False)
        self._model = None
        self._ready = False
        if build:
            self._build()

    _engine_schema = None

    @abc.abstractmethod
    def build(self):
        raise NotImplementedError('BaseJob.build')

    def _build(self):
        self.build()
        self._ready = True

    def ready(self):
        if not self._ready:
            raise RuntimeError('%s is not ready.' % str(self.__class__))

    @property
    def config(self):
        return self._config

    @property
    def model(self):
        return self._model

    @property
    @abc.abstractmethod
    def train_dataset(self):
        raise NotImplementedError('BaseJob.train_input_fn')

    @property
    @abc.abstractmethod
    def eval_dataset(self):
        raise NotImplementedError('BaseJob.eval_input_fn')

    def fit(self):
        self._model.fit(
            self.train_dataset,
            epochs=1,
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
            use_multiprocessing=False,
            **kwargs
        )
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

    def generate(self):
        tf.logging.info('Optional method `%s.generate` is not implemented. Nothing done.' % str(self.__class__))

    def present(self):
        tf.logging.info('Optional method `%s.present` is not implemented. Nothing done.' % str(self.__class__))

    def plot(self):
        tf.logging.info('Optional method `%s.plot` is not implemented. Nothing done.' % str(self.__class__))

    def history(self):
        tf.logging.info('Optional method `%s.history` is not implemented. Nothing done.' % str(self.__class__))

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
        summary_config = copy.deepcopy(self._config.run.get('summary', None))
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
