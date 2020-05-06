import json
import os
import attrdict
import utils

from forecaster.apps import base as apps_base
from forecaster.run import keys
from forecaster.run import flags


def _parse_distribute_config(config, temp_dir=None):
    distribute_config = config.run.distribute
    num_workers = len(distribute_config.workers)

    configs = []
    for i in range(num_workers):
        config_i = config.copy()
        config_i.update(
            distribute={
                'type': distribute_config.type,
                'tf_config': {
                    'cluster': {'worker': distribute_config.workers},
                    'task': {'type': 'worker', 'index': i}}})
        configs.append(config_i)
        if temp_dir is not None:
            utils.attrdict_to_json(config_i, os.path.join(temp_dir, 'config_{}'.format(i)), indent='\t')

    return configs


def run_standalone(mode, config, overwrite=False):
    """

    :param config: Distribution parsed config.
    :param overwrite:
    :param mode:
    :return:
    """
    app = apps_base.get_app(config.app)(config)
    path_config = config.path

    if mode == keys.RunningKeys.EVAL.value:
        return app.evaluate()
    if mode == keys.RunningKeys.TRAIN_EVAL.value:
        if overwrite:
            utils.fresh_dir(path_config.checkpoints_dir)
            utils.fresh_dir(path_config.tensorboard_dir)
        return app.fit()
    if mode == keys.RunningKeys.PREDICT.value:
        return app.predict()


def run_distributed(mode, config, overwrite=None):
    """
    :param config: Distribution parsed config.
    :param mode:
    :param overwrite:
    :return:
    """
    # distribute_config = _parse_distribute_config(config, temp_dir=temp_dir)

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
    if config.run.distribute.type is None:
        return run_standalone(mode, config, overwrite=overwrite)
    return run_distributed(mode, config, overwrite=overwrite)
