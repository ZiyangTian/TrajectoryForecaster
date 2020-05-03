""" . """
import forecaster


def main():
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


def data_preprocess():
    import data


if __name__ == '__main__':
    main()
