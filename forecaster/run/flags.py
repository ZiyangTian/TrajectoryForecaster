""" Run flags. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow.compat.v1 as tf

from forecaster.run import keys
from forecaster.run import monitor


def define_flags():
    """Define running flags.

    :return:
    """
    tf.flags.DEFINE_boolean('new', None, 'Create a new job.')
    tf.flags.DEFINE_string('raw_data', None, 'Path to raw data configuration file.')
    tf.flags.DEFINE_enum(
        'mode', None,
        list(keys.RunningKeys.__members__.keys()) + list(keys.PostProcessingKeys.__members__.keys()),
        'Running mode.')
    tf.flags.DEFINE_string('job_dir', None, 'Path to job directory.')
    tf.flags.DEFINE_boolean('as_monitor', True, 'Task type for distributed running.')
    tf.flags.DEFINE_string('engine', None, 'Path to engine configuration file.')
    tf.flags.DEFINE_boolean('overwrite', False, 'Overwrite or not.')

    flags = tf.flags.FLAGS
    tf.flags.mark_flags_as_mutual_exclusive(['new', 'mode'], required=True)
    tf.flags.mark_flag_as_required('job_dir')
    if flags.new:
        tf.flags.mark_flags_as_required(['raw_data', 'job_dir', 'engine'])
    return flags


def run_with_flags(flags):
    job = monitor.Monitor(flags.job_dir)
    if flags.new:
        return job.new(
            flags.engine, flags.raw_data, overwrite=flags.overwrite)
    if flags.mode is not None:
        return job.run(
            flags.mode, as_monitor=flags.as_monitor, engine_config_file=flags.engine, overwrite=flags.overwrite)

