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
from forecaster.data import sequence

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
    """Base class for jobs.
    Arguments:
        config: A `dict` like, configuration to build the job.
        warm_start_from: A `str`, representing the saved models file to warm-start from.
    """
    _engine_schema = None  # Reserved for schema.
    _schema = None

    def __init__(self, config, warm_start_from=None):
        self._config = attrdict.AttrDict(config)
        self._raw_data_spec = sequence.RawDataSpec.from_config(config.raw_data)
        self._distribute_strategy = None
        self._model = None
        self._build(warm_start_from=warm_start_from)

    @property
    @abc.abstractmethod
    def model_fn(self):
        """Build `self._model`, do not include distribute strategy here.
        Code example:
            self._model = tf.estimator.Estimator(...)
        """
        raise NotImplementedError('Job.model_fn')

    @property
    @abc.abstractmethod
    def train_input_fn(self):
        """Function which takes no argument and returns an `input_fn` for training, which can
            be used in a `tf.estimator.Estimator`. See `tf.estimator.Estimator.train`.
        """
        raise NotImplementedError('Job.train_input_fn')

    @property
    @abc.abstractmethod
    def eval_input_fn(self):
        """Function which takes no argument and returns a `input_fn` for evaluation, which can
            be used in a `tf.estimator.Estimator`. See `tf.estimator.Estimator.evaluate`.
        """
        raise NotImplementedError('Job.eval_input_fn')

    @property
    def predict_input_fn(self):
        """Function which takes no argument and return as `input_fn` for predicting, which can
            be used in an `tf.keras.Model`. See `tf.keras.Model.predict`.
        """
        tf.compat.v1.logging.error('Property `predict_dataset` is not implemented, nothing returned.')
        return None

    @property
    def config(self):
        return self._config

    @property
    def model(self):
        return self._model

    def train_eval(self):
        train_config = self._config.run.train
        train_spec = tf.estimator.TrainSpec(self.train_input_fn, max_steps=train_config.steps, hooks=None)
        eval_config = self._config.run.eval
        eval_spec = tf.estimator.EvalSpec(
            self.eval_input_fn, steps=None, name=None, hooks=None, exporters=None,
            start_delay_secs=eval_config.start_delay_seconds, throttle_secs=eval_config.throttle_secs)
        return tf.estimator.train_and_evaluate(self._model, train_spec, eval_spec)

    def evaluate(self):
        # eval_config = self._config.run.eval
        return self._model.evaluate(
            self.eval_input_fn, steps=None, hooks=None)

    def predict(self):
        if self.predict_input_fn is None:
            raise NotImplementedError('Job.predict_input_fn')
        # predict_config = self._config.run.predict
        self._model.predict(
            self.predict_input_fn, predict_keys=None, hooks=None, checkpoint_path=None)

    def _build(self, warm_start_from=None):
        save_config = self._config.run.save
        run_config = tf.estimator.RunConfig(
            save_summary_steps=save_config.save_summary_steps,
            save_checkpoints_steps=save_config.save_checkpoints_steps,
            keep_checkpoint_max=save_config.keep_checkpoint_max)
        distribute_config = self._config.run.distribute
        distribute_type = distribute_config.type

        if distribute_type is not None:
            communication = getattr(tf.distribute.experimental.CollectiveCommunication, distribute_type.upper())
            tf_config = distribute_config.tf_config
            os.environ['TF_CONFIG'] = json.dumps(dict(tf_config))
            self._distribute_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(communication)
            run_config.replace(train_distribute=self._distribute_strategy)

        self._model = tf.estimator.Estimator(
            self.model_fn, model_dir=self._config.run.save.model_dir,
            config=run_config, params=None, warm_start_from=warm_start_from)
