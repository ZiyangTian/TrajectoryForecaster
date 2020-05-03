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
    app.train_eval()


if __name__ == '__main__':
    main()
