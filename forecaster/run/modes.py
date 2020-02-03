""" Running modes. """
import os
import shutil

from forecaster.version import __version__
from forecaster.apps import base as apps_base
from utils import os as os_utils, typing as typing_utils


def _indexed_file_name_path(dir_path, file_name, suffix=None, get_latest=False):
    existed_file_names = os.listdir(dir_path)
    suffix = '' if suffix is None else suffix

    max_index = -1
    for name in filter(lambda f: f.startswith(file_name) and f.endswith(suffix), existed_file_names):
        index = int(name[:- len(suffix)].split('-')[-1])
        if index > max_index:
            max_index = index
    if not get_latest:
        max_index += 1
    return os.path.join(dir_path, '%s-%i%s' % (file_name, max_index, suffix))


def _allocate_job_dir(job_dir, overwrite=False):
    if os.path.exists(job_dir) and overwrite:
        shutil.rmtree(job_dir)
    if not os.path.exists(job_dir):
        os_utils.make_dir(job_dir)
    os_utils.make_dir(os.path.join(job_dir, 'checkpoints'))
    os_utils.make_dir(os.path.join(job_dir, 'presentation'))
    os_utils.make_dir(os.path.join(job_dir, 'configs'))
    os_utils.make_dir(os.path.join(job_dir, 'logs'))
    os_utils.make_dir(os.path.join(job_dir, 'temp'))
    with open(os.path.join(job_dir, '__jobdir__'), 'w') as f:
        f.write('version=%s' % __version__)
    return job_dir


def _load_latest_config(job_dir):
    config_file = _indexed_file_name_path(
        os.path.join(job_dir, 'configs'), 'config', '.json', get_latest=True)
    if not os.path.exists(config_file):
        raise FileNotFoundError('Config file not found.')
    return typing_utils.attrdict_from_json(config_file)


def get_job(config, build=False):
    return apps_base.REGISTERED_APPS[config.task](config, build=build)


def new(job_dir, engine_config_file, raw_data_config_file, overwrite=True):
    # TODO: add schema validation.
    engine_config = typing_utils.attrdict_from_json(engine_config_file)
    raw_data_config = typing_utils.attrdict_from_json(raw_data_config_file)
    config = apps_base.REGISTERED_APPS[engine_config.task].parse_configs(
        job_dir, engine_config, raw_data_config)

    _allocate_job_dir(job_dir, overwrite=overwrite)
    config_file_path = _indexed_file_name_path(
        os.path.join(config.run.job_dir, 'configs'), 'config', '.json')
    typing_utils.attrdict_to_json(config, config_file_path, indent='\t')


def _maybe_new_config(job_dir, engine_config_file=None):
    config = _load_latest_config(job_dir)
    if engine_config_file:
        config = parse_configs(
            config.run.job_dir,
            typing_utils.attrdict_from_json(engine_config_file),
            config.raw_data)
        config_file = _indexed_file_name_path(
            os.path.join(job_dir, 'configs'), 'config', '.json', get_latest=False)
        typing_utils.attrdict_to_json(config, config_file, indent='\t')
    return config


def train(config):
    job = get_job(config, build=True)
    job.train()


def evaluate(config):
    pass


