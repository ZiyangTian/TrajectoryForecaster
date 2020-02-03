""" . """
from forecaster.run import flags
from forecaster.run import distributed
from forecaster.run import standalone


def run_with_flags():
    flags.define_flags()
    distributed.run()


def main():
    pass


if __name__ == '__main__':
    main()
