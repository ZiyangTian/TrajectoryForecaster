import copy
import sys
import os
import json
import subprocess

import tensorflow as tf

from threading import Event
from forecaster.run import keys
from forecaster.run import flags as run_flags
from forecaster.run import modes
from forecaster.run import standalone
from forecaster.run import manager
from utils import typing as typing_utils


_TERMINATION_CHECKING_INTERVAL = 3


def get_strategy(name):
    if hasattr(tf.distribute, name):
        return getattr(tf.distribute, name)
    elif hasattr(tf.distribute.experimental, name):
        return getattr(tf.distribute.experimental, name)
    raise ValueError('Unexpected strategy name: %s.' % name)


def assign_roles(mode, strategy, ports):
    """ Generate cluster configuration, task type and index.
        Arguments:
            mode: A `RunningKeys` key.
            strategy: A `tf.distribute.Strategy` instance.
            ports: A `dict`, map from a string port ID to a `dict` of devices which
                follows the signature:
                    {"CPUs": [<cpu_device>, ...], "GPUs": [<gpu_device>, ...]}
        Returns:
            A `list` of `TF_CONFIG`s.
    """
    # TODO: add multi-device support.
    port_ids = list(ports.keys())
    num_ports = len(port_ids)
    tf_configs = []
    if strategy == tf.distribute.experimental.MultiWorkerMirroredStrategy:
        if mode == keys.RunningKeys.fit:
            if num_ports <= 1:
                pass  # TODO:
            else:
                cluster = {'worker': port_ids[0: -1], 'evaluator': port_ids[-1]}
                for i in range(num_ports):
                    if i == num_ports - 1:
                        task = {'type': 'evaluator', 'index': 0}
                    else:
                        task = {'type': 'worker', 'index': i}
                    tf_configs.append({'cluster': cluster, 'task': task})
        else:
            cluster = {'worker': port_ids}
            for i in range(num_ports):
                tf_configs.append({'cluster': cluster, 'task': {'type': 'worker', 'index': i}})
    elif strategy == tf.distribute.experimental.CentralStorageStrategy:
        pass
    else:
        raise ValueError('Unexpected distribute strategy: %s.' % str(strategy))
    return tf_configs


def run_on_one_worker(raw_config, cluster, task_type, task_index):
    """

    :param raw_config:
    :param task_type: A `str`,
    :param task_index: An `int`
    :return:
    """
    config = copy.deepcopy(raw_config)
    config.tf_config = {'cluster': cluster, 'task': {'type': task_type, 'index': task_index}}
    typing_utils.attrdict_to_json(config)

    raw_config.run.distribute.update(task={'type': task_type, 'index': task_index})
    job = modes.get_job(raw_config, build=True)
    job.train_eval()


def assign_role(strategy, ports):
    port_ids = list(ports.keys())
    num_ports = len(port_ids)

    types, indexes, = [], []
    if strategy == tf.distribute.experimental.MultiWorkerMirroredStrategy:
        if num_ports <= 1:
            pass  # TODO:
        else:
            cluster = {'worker': port_ids[0: -1], 'evaluator': port_ids[-1]}
            for i in range(num_ports):
                if i == num_ports - 1:
                    task = {'type': 'evaluator', 'index': 0}
                else:
                    task = {'type': 'worker', 'index': i}
                tf_configs.append({'cluster': cluster, 'task': task})
    elif strategy == tf.distribute.experimental.CentralStorageStrategy:
        pass
    else:
        raise ValueError('Unexpected distribute strategy: %s.' % str(strategy))
    return cluster, types, indexes


def terminate_all(processes):
    for p in processes:
        try:
            p.terminate()
        except Exception as e:
            tf.logging.error('Terminating process %s failed with error %s.' % (str(p), str(e)))
        tf.logging.info('Successfully terminated process %s.' % str(p))
    sys.exit(0)


def run_on_multi_workers(config, job_dir):
    mng = manager.Manager(job_dir)
    config = copy.deepcopy(config)
    distribute_config = config.run.pop('distribute')
    strategy = get_strategy(distribute_config.strategy)
    ports = distribute_config.ports
    num_workers = len(ports)
    cluster, types, indexes = assign_role(strategy, ports)

    processes = []
    for i in range(num_workers):
        processes.append(run_on_one_worker(...))

    termination_event = Event()
    while True:
        termination_event.wait(_TERMINATION_CHECKING_INTERVAL)
        if os.path.exists(mng.description_file):
            with open(mng.description_file, 'r') as f:
                content = f.readlines()
                #if mng.


def main():
    run_flags.define_flags_for_distribution()
    run_flags.define_flags()
    flags = tf.flags.FLAGS

    run_on_one_worker(flags.mode, config)



if __name__ == '__main__':
    main()
