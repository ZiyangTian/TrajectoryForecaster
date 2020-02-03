""" Data type utilities. """
import json
import collections
import attrdict


def attrdict_from_json(file):
    with open(file, 'r') as f:
        obj = json.load(f)
    if type(obj) is dict:
        return attrdict.AttrDict(obj)
    raise ValueError('Unexpected json content in file: %s.' % file)


def attrdict_to_json(obj, file, **kwargs):
    with open(file, 'w') as f:
        json.dump(dict(obj), f, **kwargs)


def normalize_list_of_type(obj, allowed_type, allow_empty=False, allow_duplicated=False):
    """ Normalize a list of objects of a specified type. Converts one or more object(s)
            to a python list.
        Arguments:
            obj: A single or an container of object(s) of type `allowed_type`.
            allowed_type: A `type`, allowed type of `obj`.
            allow_empty: A `bool`, whether empty object is allowed, defaults to false.
            allow_duplicated: A `bool`, whether duplicated elements are allowed, defaults to false.
        Returns:
            A `list`.
        Raises:
            ValueError: For any invalid inputs, such as empty, invalid types, etc.
    """
    if not obj:
        if allow_empty:
            return []
        else:
            raise ValueError('Empty object is not allowed, got %s.' % str(obj))
    if isinstance(obj, allowed_type):
        return [obj]
    if isinstance(obj, collections.Iterable):
        obj_list = []
        for o in obj:
            if not isinstance(o, allowed_type):
                raise ValueError('Unexpected data type, got input %s and type %s.' % (str(o), str(type(o))))
            if not allow_duplicated and o in obj_list:
                raise ValueError('Found duplicated elements: %s.' % str(o))
            obj_list.append(o)
        return obj_list
    raise ValueError('Unexpected value type, got %s.' % type(obj))
