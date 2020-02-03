import os
import shutil

from forecaster import version
from utils import os as os_utils
from utils import typing as typing_utils


class Manager(object):
    def __init__(self, job_dir):
        self._job_dir = job_dir

    _SHOULD_TERMINATE_FLAG = 'should_terminate'

    @property
    def config_dir(self):
        return os.path.join(self._job_dir, 'configs')

    @property
    def log_dir(self):
        return os.path.join(self._job_dir, 'logs')

    @property
    def temp_dir(self):
        return os.path.join(self._job_dir, 'temp')

    @property
    def presentation_dir(self):
        return os.path.join(self._job_dir, 'presentation')

    @property
    def checkpoints_dir(self):
        return os.path.join(self._job_dir, 'checkpoints')

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
            os_utils.make_dir(self._job_dir)
        os_utils.make_dir(self.config_dir)
        os_utils.make_dir(self.log_dir)
        os_utils.make_dir(self.temp_dir)
        os_utils.make_dir(self.presentation_dir)
        os_utils.make_dir(self.checkpoints_dir)
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
        return typing_utils.attrdict_from_json(config_file)
