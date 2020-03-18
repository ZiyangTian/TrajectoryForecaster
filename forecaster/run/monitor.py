import os
import json
import shutil
import utils

from forecaster import version
from forecaster.run import distribute


class Monitor(object):

    _SHOULD_TERMINATE_FLAG = 'should_terminate'

    def __init__(self, job_dir):
        self._job_dir = job_dir
        self._config_dir = os.path.join(self._job_dir, 'configs')
        self._log_dir = os.path.join(self._job_dir, 'logs')
        self._temp_dir = os.path.join(self._job_dir, 'temp')
        self._presentation_dir = os.path.join(self._job_dir, 'presentation')
        self._checkpoints_dir = os.path.join(self._job_dir, 'checkpoints')

    @property
    def description_file(self):
        return os.path.join(self._job_dir, '__jobdir__')

    def job_exists(self):
        return os.path.exists(self._job_dir) and os.path.exists(self.description_file)

    def should_terminate(self):
        if self.job_exists():
            with open(self.description_file, 'r') as f:
                content = f.readlines()
                if self._SHOULD_TERMINATE_FLAG in content:
                    return True
        return False
        
    def allocate_dirs(self, overwrite=True):
        if os.path.exists(self._job_dir) and overwrite:
            shutil.rmtree(self._job_dir)
        if not os.path.exists(self._job_dir):
            utils.os.make_dir(self._job_dir)
        utils.os.make_dir(self._config_dir)
        utils.os.make_dir(self._log_dir)
        utils.os.make_dir(self._temp_dir)
        utils.os.make_dir(self._presentation_dir)
        utils.os.make_dir(self._checkpoints_dir)
        with open(self.description_file, 'w') as f:
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
            os.path.join(self._job_dir, 'configs'), 'config', '.json', get_latest=True)
        if not os.path.exists(config_file):
            raise FileNotFoundError('Configuration file not found.')
        return utils.typing.attrdict_from_json(config_file)

    def _parse_distribute_config(self, config):
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

            with open(os.path.join(self._temp_dir, 'config_{}'.format(i)), 'w') as f:
                json.dump(config_i, f)

        return configs

    def new(self, engine_config_file, raw_data_config_file, overwrite=True):
        config = utils.typing.attrdict_from_json(engine_config_file)
        config.update(
            path={
                'job_dir': self._job_dir,
                'config_dir': self._config_dir,
                'log_dir': self._log_dir,
                'temp_dir': self._temp_dir,
                'presentation_dir': self._presentation_dir,
                'checkpoints_dir': self._checkpoints_dir},
            raw_data=dict(utils.typing.attrdict_from_json(raw_data_config_file)))
        self.allocate_dirs(overwrite=overwrite)
        config_file_path = self._indexed_file_name_path(
            self._config_dir, 'config', '.json')
        utils.typing.attrdict_to_json(config, config_file_path, indent='\t')

    def run(self, mode, as_monitor=True, engine_config_file=None, overwrite=False):
        if as_monitor:
            config = self._load_latest_config()
            if engine_config_file is not None:
                config.update(dict(utils.typing.attrdict_from_json(engine_config_file)))
                config_file_path = self._indexed_file_name_path(
                    self._config_dir, 'config', '.json')
                utils.typing.attrdict_to_json(config, config_file_path, indent='\t')
            return distribute.run_as_monitor(mode, config, overwrite=overwrite)

        if engine_config_file is None:
            raise ValueError('`engine_config_file` must be specified if run as non-monitor.')
        return distribute.run_standalone(mode, utils.typing.attrdict_from_json(engine_config_file), overwrite=False)
