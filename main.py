""" . """
import forecaster


def run_flags():
    # flags = run_flags.define_flags()

    import attrdict
    flags = attrdict.AttrDict()
    flags.new = 0
    if flags.new:
        flags.job_dir = 'saved/fake'
        flags.engine = 'engines/engine.json'
        flags.raw_data = 'engines/data.json'
        flags.overwrite = True
    else:
        flags.mode = 'train_eval'
        flags.job_dir = 'saved/fake'
        flags.as_monitor = True  # Do not change.
        flags.engine = 'engines/engine.json'
        flags.overwrite = True

    forecaster.run_with_flags(flags)


def main():
    import utils
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    raw_data_config = utils.attrdict_from_json('engines/data.json')
    config = utils.attrdict_from_json('engines/engine_detection.json')
    config.update(raw_data=raw_data_config)
    utils.attrdict_to_json(config, 'engines/config.json', indent='\t')

    app = forecaster.get_app(config.app)(config)
    app.fit()


def monitor_test():
    monitor = forecaster.Monitor('/Users/Tianziyang/projects/saved/job_1')

    monitor.new(
        engine_config_file='engines/engine_prediction.json',
        raw_data_config_file='engines/data.json',
        overwrite=True)

    monitor.run(
        forecaster.RunningKeys.TRAIN_EVAL.value,
        engine_config_file='engines/engine_prediction.json',
        overwrite=True)


def main1():
    import tensorflow as tf
    import forecaster
    dataset1 = tf.data.Dataset.from_tensor_slices(
        {
            'a': list(float(i) for i in range(10)),
            'b': list(float(i) for i in range(10, 20)),
            'c': [0.1 * i for i in range(10)]})
    dataset2 = tf.data.Dataset.from_tensor_slices(
        {
            'a': list(float(i) for i in range(10, 20)),
            'b': list(float(i) for i in range(20, 30)),
            'c': [1. + 0.1 * i for i in range(10)]})
    datasets = tf.data.Dataset.from_tensor_slices([1, 2]).map(lambda x: dataset1 if x == 1 else dataset2)
    cs1 = forecaster.SequentialColumnsSpec(['a', 'c'], 4, new_columns='cs1')
    cs2 = forecaster.ReducingColumnsSpec(['a'], 'mean', 1, 3, new_columns='cs2')
    cs3 = forecaster.ReservingColumnSpec(['b'], 1, new_columns='cs3')
    sequencer = forecaster.Sequencer([cs1, cs2, cs3], datasets)
    dataset = sequencer.sequence_dataset(1, 2, 2, 2)
    for x in dataset:
        tf.print(x)


def main():
    import functools
    import glob

    data_files = glob.glob(r'E:\S11\3_trjectories_prdict_room16\data_TrajectoryForecaster\test\*.txt')
    columns = ['t', 'x', 'y', 'z', 'xt', 'yt', 'zt']
    defaults = [0., 0., 0., 0., 0., 0., 0.]

    datasets = tf.data.Dataset.from_tensor_slices(data_files)  # 文件名 -> 数据集
    datasets = datasets.map(
        functools.partial(tf.data.experimental.CsvDataset, record_defaults=defaults, header=False),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)  # 载入文件内容
    datasets = datasets.map(
        functools.partial(dataset_utils.named_dataset, names=columns),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)  # 加入特征名
    columns_specs = [
        SequenceColumnsSpec(['t', 'x'], 5, group=True, new_names='c1'),
        SequenceColumnsSpec(['t', 'x'], 5, offset=5, group=True, new_names='c2'),
    ]
    dataset = make_sequential_dataset(
        datasets,
        columns_specs,
        shift=2,
        stride=3,
        shuffle_buffer_size=None,
        name=None)

    for x in dataset:
        tf.print(x)


if __name__ == '__main__':
    main1()
