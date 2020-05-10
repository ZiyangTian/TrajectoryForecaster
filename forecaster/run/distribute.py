import os
import subprocess
import utils

from forecaster.apps import base as apps_base
from forecaster.run import keys


def _parse_distribute_config(config, temp_dir=None):
    distribute_config = config.run.distribute
    num_workers = len(distribute_config.workers)

    configs = []
    config_files = []
    for i in range(num_workers):
        config_i = config.copy()
        config_i.run.update(
            distribute={
                'type': distribute_config.type,
                'tf_config': {
                    'cluster': {'worker': distribute_config.workers},
                    'task': {'type': 'worker', 'index': i}}})
        configs.append(config_i)
        if temp_dir is not None:
            config_file = os.path.join(temp_dir, 'config_{}'.format(i))
            utils.attrdict_to_json(config_i, config_file, indent='\t')
            config_files.append(config_file)

    return configs if temp_dir is None else (configs, config_files)


def run_standalone(mode, config):
    """

    :param config: Distribution parsed config.
    :param mode:
    :return:
    """
    app = apps_base.get_app(config.app)(config)

    if mode == keys.RunningKeys.EVAL.value:
        return app.evaluate()
    if mode == keys.RunningKeys.TRAIN_EVAL.value:
        return app.fit()
    if mode == keys.RunningKeys.PREDICT.value:
        return app.predict()


def run_distributed(mode, config):
    """
    :param config: Distribution parsed config.
    :param mode:
    :return:
    """
    # distribute_config = _parse_distribute_config(config, temp_dir=temp_dir)

    if mode != keys.RunningKeys.train_eval:
        raise ValueError('Only `train_eval` mode is supported.')

    temp_dir = config.path.temp_dir
    _, config_files = _parse_distribute_config(config, temp_dir=temp_dir)

    processes = []
    for i, config_file in enumerate(config_files):
        command = [
            'python',
            os.path.join('forecaster', 'run', 'distribute_entry.py'),
            '--job_dir={}'.format(config.path.job_dir),
            '--mode=train_eval',
            'engine={}'.format(config_file),
            '--as_monitor=False',
            '>',
            os.path.join(config.path.log_dir, 'log_{}.log'.format(i))]
        processes.append(subprocess.Popen(command, shell=True))
    return processes


def run_as_monitor(mode, config, overwrite=None):
    """
    :param mode:
    :param config: Distribution unparsed config.
    :param overwrite:
    :return:
    """
    if overwrite:
        if mode == keys.RunningKeys.train_eval.value:
            path_config = config.path
            utils.fresh_dir(path_config.checkpoints_dir)
            utils.fresh_dir(path_config.tensorboard_dir)
            utils.fresh_dir(path_config.log_dir)
    if config.run.distribute.type is None:
        return run_standalone(mode, config)
    return run_distributed(mode, config)
