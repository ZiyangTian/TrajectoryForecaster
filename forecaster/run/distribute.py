import json
import os
import attrdict
import utils

from forecaster.apps import base as apps_base
from forecaster.run import keys
from forecaster.run import flags


def run_standalone(mode, config, overwrite=False):
    """

    :param config: Distribution parsed config.
    :param overwrite:
    :param mode:
    :return:
    """
    app = apps_base.REGISTERED_APPS[config.app](config)

    if overwrite:
        utils.fresh_dir(config.path.checkpoints_dir)
    if mode == keys.RunningKeys.TRAIN.value:
        return app.train()
    if mode == keys.RunningKeys.EVAL.value:
        return app.eval()
    if mode == keys.RunningKeys.TRAIN_EVAL.value:
        return app.train_eval()
    if mode == keys.RunningKeys.PREDICT.value:
        return app.predict()


def run_distributed(mode, config, overwrite=False):
    """
    :param config: Distribution parsed config.
    :param mode:
    :param overwrite:
    :return:
    """
    if overwrite:
        ...

    run_standalone(mode, config, overwrite=False)


def run_as_monitor(mode, config, overwrite=None, temp_dir=None):
    """
    :param mode:
    :param config: Distribution unparsed config.
    :param overwrite:
    :param temp_dir:
    :return:
    """
    configs = _parse_distribute_config(config, temp_dir=temp_dir)

    if config.run.distribute.type is None:
        return run_standalone(mode, next(iter(configs)), overwrite=overwrite)
    return run_distributed(mode, config, overwrite=overwrite)
