""" Running keys. """
import enum


@enum.unique
class RunningKeys(enum.Enum):
    train = 'train'
    evaluate = 'evaluate'
    fit = 'fit'
    predict = 'predict'


@enum.unique
class PostProcessingKeys(enum.Enum):
    history = 'history'
    present = 'present'
    plot = 'plot'


