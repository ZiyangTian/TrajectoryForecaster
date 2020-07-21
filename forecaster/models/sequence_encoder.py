"""Sequence encoder class."""
import functools
import tensorflow as tf

from forecaster.ops import diff
from forecaster.models import layers
from forecaster.models import networks


