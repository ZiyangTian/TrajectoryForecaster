import abc
import os
import json
import shutil
import utils
import attrdict
import tensorflow as tf

from forecaster import version
from forecaster.data import sequence
from forecaster.run import distribute
from forecaster.run import flags


class Monitor(object):
    _SHOULD_TERMINATE_FLAG = 'should_terminate'

    def __init__(self, job_dir):
        self._job_dir = job_dir
        self._config_dir = os.path.join(self._job_dir, 'configs')
        self._log_dir = os.path.join(self._job_dir, 'logs')
        self._temp_dir = os.path.join(self._job_dir, 'temp')
        self._presentation_dir = os.path.join(self._job_dir, 'presentation')
        self._checkpoints_dir = os.path.join(self._job_dir, 'checkpoints')
        self._tensorboard_dir = os.path.join(self._job_dir, 'tensorboard')
        self._description_file = os.path.join(self._job_dir, '__jobdir__')

    def job_exists(self):
        return os.path.exists(self._job_dir) and os.path.exists(self._description_file)

    def should_terminate(self):
        if self.job_exists():
            with open(self._description_file, 'r') as f:
                content = f.readlines()
                return self._SHOULD_TERMINATE_FLAG in content
        return False
        
    def allocate_dirs(self, overwrite=True):
        if self.job_exists() and overwrite or not self.job_exists():
            utils.fresh_dir(self._job_dir)
            utils.make_dir(self._config_dir)
            utils.make_dir(self._log_dir)
            utils.make_dir(self._temp_dir)
            utils.make_dir(self._presentation_dir)
            utils.make_dir(self._checkpoints_dir)
            utils.make_dir(self._tensorboard_dir)
            with open(self._description_file, 'w') as f:
                f.write('version=%s' % version.__version__)

    @staticmethod
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

    def _load_latest_config(self):
        config_file = self._indexed_file_name_path(
            self._config_dir, 'config', '.json', get_latest=True)
        if not os.path.exists(config_file):
            raise FileNotFoundError('Configuration file not found.')
        return utils.attrdict_from_json(config_file)

    def new(self, engine_config_file, raw_data_config_file, overwrite=True):
        config = utils.attrdict_from_json(engine_config_file)
        config.update(
            path={
                'job_dir': self._job_dir,
                'config_dir': self._config_dir,
                'log_dir': self._log_dir,
                'temp_dir': self._temp_dir,
                'presentation_dir': self._presentation_dir,
                'checkpoints_dir': self._checkpoints_dir,
                'tensorboard_dir': self._tensorboard_dir},
            raw_data=utils.attrdict_from_json(raw_data_config_file))
        self.allocate_dirs(overwrite=overwrite)
        config_file_path = self._indexed_file_name_path(
            self._config_dir, 'config', '.json')
        utils.attrdict_to_json(config, config_file_path, indent='\t')

    def run(self, mode, as_monitor=True, engine_config_file=None, overwrite=False):
        if as_monitor:
            config = self._load_latest_config()
            if engine_config_file is not None:
                config.update(dict(utils.attrdict_from_json(engine_config_file)))
                config_file_path = self._indexed_file_name_path(
                    self._config_dir, 'config', '.json')
                utils.attrdict_to_json(config, config_file_path, indent='\t')
            return distribute.run_as_monitor(mode, config, overwrite=overwrite)

        if engine_config_file is None:
            raise ValueError('`engine_config_file` must be specified if run as non-monitor.')
        return distribute.run_standalone(mode, utils.attrdict_from_json(engine_config_file), overwrite=False)

    @classmethod
    def run_with_flags(cls):
        flags.define_flags()
        flags_obj = flags.parse_flags()

        job = cls(flags_obj.job_dir)
        if flags_obj.new:
            return job.new(
                flags_obj.engine,
                flags_obj.raw_data,
                overwrite=flags_obj.overwrite)
        return job.run(
            flags_obj.mode,
            as_monitor=flags_obj.as_monitor,
            engine_config_file=flags_obj.engine,
            overwrite=flags_obj.overwrite)
