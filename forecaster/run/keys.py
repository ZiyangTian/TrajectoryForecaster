""" Running keys. """
import enum


@enum.unique
class RunningKeys(enum.Enum):
    TRAIN = 'train'
    EVAL = 'evaluate'
    TRAIN_EVAL = 'train_eval'  # For keras running.
    PREDICT = 'predict'


@enum.unique
class PostProcessingKeys(enum.Enum):
    history = 'history'
    present = 'present'
    plot = 'plot'


