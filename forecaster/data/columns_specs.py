""" Columns Specs for creating dataset sequencer. """
import abc
import tensorflow as tf

from utils import typing as typing_utils


class ColumnsSpec(object):
    """Column pre-processing spec base class. """
    def __init__(self, raw_columns, group=True, new_columns=None, name=None):
        self._raw_columns = typing_utils.normalize_list_of_type(raw_columns, str)
        self._group = group
        self._new_columns = self._allocate_new_names(new_columns)
        self._name = name or 'ColumnSpec'

    @abc.abstractmethod
    def call(self, sequence):
        """Convert a sequence tensor for creating the new dataset.
            Arguments:
                sequence: A `Tensor` of shape (`self.max_offset`, ...).
            Returns:
                A converted `Tensor` of determined shape.
        """
        raise NotImplementedError('ColumnSpec.call')

    @property
    @abc.abstractmethod
    def max_offset(self):
        raise NotImplementedError('ColumnsSpec.max_offset')

    @property
    def raw_columns(self):
        return self._raw_columns

    @property
    def new_columns(self):
        return self._new_columns

    @property
    def name(self):
        return self._name

    def __call__(self, windowed_dict):
        """Map from a `dict` of windowed feature tensors to converted feature tensors.
            Arguments:
                windowed_dict: A `dict` from `str` feature names to feature `Tensor`s.
            Returns:
                A `dict` from `str` feature names to converted feature `Tensor`s.
        """
        with tf.name_scope('%s_call' % self._name):
            processed_features = map(lambda k: self.call(windowed_dict[k]), self._raw_columns)
            if self._group:
                processed_features = {self._new_columns: tf.stack(list(processed_features), axis=-1)}
            else:
                processed_features = dict(zip(self._new_columns, processed_features))
        return processed_features

    def get_new_names(self, raw_name):
        if self._group:
            raise AttributeError('Cannot `get_new_names` when `group` is true.')
        return self._new_columns[self._raw_columns.index(raw_name)]

    def _allocate_new_names(self, new_names=None):
        if self._group:
            if new_names is None:
                return '%s___%s' % (self.name, '__'.join(self._raw_columns) % type(new_names))
            if type(new_names) is not str:
                raise ValueError('`new_columns` must be a `str` if `group` is true, got %s.' % new_names)
            return new_names

        if new_names is None:
            return list(map(lambda n: '%s___%s' % (self.name, n), self._raw_columns))

        new_names = typing_utils.normalize_list_of_type(new_names, str)
        if len(new_names) != len(self._raw_columns):
            raise ValueError('Number of `new_columns` elements must be equal to `raw_columns`, '
                             'got %i and %i.' % (len(new_names), len(self._raw_columns)))
        return new_names


class SequentialColumnsSpec(ColumnsSpec):
    """Sequential columns spec.
        Arguments:
            raw_columns: A `list` of `str`, raw column names.
            sequence_length: An `int`, sequence length.
            offset: An `int`, relative offset of sequence starting position.
            group: A `bool`, whether to group columns. If true, all raw columns must have
                identical data type and tensor shape.
            new_columns: A (sequence of) `str`, new column name(s). If group is true, `new_columns`
                must be a `str`; or else, its elements must match the order of `raw_columns`.
                By default, automatically generate.
            name: A `str`, name of the columns spec.
        Raises:
            ValueError: If `new_columns` is not properly specified.
    """
    def __init__(self, raw_columns, sequence_length, offset=0, group=True, new_columns=None, name=None):
        self._sequence_length = sequence_length
        self._offset = offset
        super(SequentialColumnsSpec, self).__init__(
            raw_columns, group=group, new_columns=new_columns, name=name or 'SequentialColumnsSpec')

    @property
    def max_offset(self):
        return self._sequence_length + self._offset

    def call(self, sequence):
        return sequence[self._offset: self._offset + self._sequence_length]


class ReducingColumnsSpec(ColumnsSpec):
    """Reducing columns spec.
        Arguments:
            raw_columns: A `list` of `str`, raw column names.
            reduction: A callable, representing the reduction function.
            begin: An `int`, the beginning position.
            end: An `int`, the ending position.
            group: A `bool`, whether to group columns. If true, all raw columns must have
                identical data type and tensor shape.
            new_columns: A (sequence of) `str`, new column name(s). If group is true, `new_columns`
                must be a `str`; or else, its elements must match the order of `raw_columns`.
                By default, automatically generate.
            name: A `str`, name of the columns spec.
        Raises:
            ValueError: If `new_columns` is not properly specified.
    """
    def __init__(self, raw_columns,
                 reduction=None, begin=None, end=None,
                 group=True, new_columns=None, name=None):
        self._reduction_fn = self._get_reduction_fn(reduction)
        self._begin = 0 if begin is None else begin
        self._end = self._begin + 1 if begin is None else end
        super(ReducingColumnsSpec, self).__init__(
            raw_columns, group=group, new_columns=new_columns, name=name or 'ReducingColumnsSpec')

    @property
    def max_offset(self):
        return self._end

    def call(self, sequence):
        return self._reduction_fn(sequence[self._begin: self._end])

    @staticmethod
    def _get_reduction_fn(reduction):
        if callable(reduction):
            return reduction
        identification = reduction.lower()

        if identification == 'sum':
            return tf.reduce_sum
        if identification in {'average', 'mean'}:
            return tf.reduce_mean
        if identification in {'maximum', 'max'}:
            return tf.reduce_max
        if identification in {'minimum', 'min'}:
            return tf.reduce_min
        if identification == 'all':
            return tf.reduce_all
        if identification == 'any':
            return tf.reduce_any
        if identification in {'prod', 'product'}:
            return tf.reduce_prod


class ReservingColumnSpec(ColumnsSpec):
    """Reserving columns spec.
        Arguments:
            raw_columns: A `list` of `str`, raw column names.
            position: A `int`, reserving position.
            group: A `bool`, whether to group columns. If true, all raw columns must have
                identical data type.
            new_columns: A (sequence of) `str`, new column name(s). If group is true, `new_columns`
                must be a `str`; or else, its elements must match the order of `raw_columns`.
                By default, automatically generate.
            name: A `str`, name of the columns spec.
        Raises:
            ValueError: If `new_columns` is not properly specified.
    """
    def __init__(self, raw_columns, position=0, group=True, new_columns=None, name=None):
        self._position = position
        super(ColumnsSpec, self).__init__(
            raw_columns, group=group, new_names=new_columns, name=name or 'ReservingColumnSpec')

    @property
    def max_offset(self):
        return self._position + 1

    def call(self, sequence):
        return sequence[self._position]
