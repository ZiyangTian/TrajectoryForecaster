""" Run flags. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import attrdict
import tensorflow.compat.v1 as tf

from forecaster.run import keys


def define_flags():
    """Define running flags.
    Create a new job:
    ```
        python <main.py> \
            --new \
            --job_dir=<path/to/job_dir> \
            --engine=<path/to/engine_config_file> \
            --raw_data=<path/to/raw_data_config_file> \
            --overwrite (optional)
    ```
    Run an existed job:
    ```
        python <main.py> \
            --job_dir=<path/to/job_dir> \
            --mode=<mode> \
            --engine=<path/to/engine_config_file> (optional)
            --overwrite (optional)
    ```

    :return:
    """
    tf.flags.DEFINE_boolean('new', None, 'Create a new job.')
    tf.flags.DEFINE_string('raw_data', None, 'Path to raw data configuration file.')
    tf.flags.DEFINE_enum(
        'mode', None,
        list(keys.RunningKeys.__members__.keys()) + list(keys.PostProcessingKeys.__members__.keys()),
        'Running mode.')
    tf.flags.DEFINE_string('job_dir', None, 'Path to job directory.')
    tf.flags.DEFINE_string('engine', None, 'Path to engine configuration file.')
    tf.flags.DEFINE_boolean('overwrite', False, 'Overwrite or not.')

    tf.flags.DEFINE_boolean('as_monitor', True, 'Task type for distributed running.')

    flags = tf.flags.FLAGS
    tf.flags.mark_flags_as_mutual_exclusive(['new', 'mode'], required=True)
    tf.flags.mark_flag_as_required('job_dir')
    if flags.new:
        tf.flags.mark_flags_as_required(['raw_data', 'job_dir', 'engine'])
    return flags


def parse_flags():
    flags = tf.flags.FLAGS
    flags_dict = {
        'job_dir': flags.job_dir,
        'engine': flags.engine,
        'overwrite': flags.overwrite}
    if flags.new:
        flags_dict.update(type='new', raw_data=flags.raw_data)
    flags_dict.update(type='run', mode=flags.mode, overwrite=flags.overwrite, as_monitor=flags.as_monitor)
    return attrdict.AttrDict(flags_dict)
