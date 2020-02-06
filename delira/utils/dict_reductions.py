from collections import MutableMapping
from typing import Union, Dict, Callable
import numpy as np


# Reduction Functions
def reduce_last(items: list) -> Union[float, int, np.ndarray]:
    """
    Reduction Function returning the last element

    Parameters
    ----------
    items : list
        the items to reduce

    Returns
    -------
    float, int or :class:`numpy.ndarray`
        reduced items

    """
    return items[-1]


def reduce_first(items: list) -> Union[float, int, np.ndarray]:
    """
    Reduction Function returning the first element

    Parameters
    ----------
    items : list
        the items to reduce

    Returns
    -------
    float, int or :class:`numpy.ndarray`
        reduced items

    """
    return items[0]


def reduce_mean(items: list) -> Union[float, int, np.ndarray]:
    """
    Reduction Function returning the mean element

    Parameters
    ----------
    items : list
        the items to reduce

    Returns
    -------
    float, int or :class:`numpy.ndarray`
        reduced items

    """
    return np.mean(items)


def reduce_median(items: list) -> Union[float, int, np.ndarray]:
    """
    Reduction Function returning the median element

    Parameters
    ----------
    items : list
        the items to reduce

    Returns
    -------
    float, int or :class:`numpy.ndarray`
        reduced items

    """
    return np.median(items)


def reduce_max(items: list) -> Union[float, int, np.ndarray]:
    """
    Reduction Function returning the max element

    Parameters
    ----------
    items : list
        the items to reduce

    Returns
    -------
    float, int or :class:`numpy.ndarray`
        reduced items

    """
    return np.max(items)


def reduce_min(items: list) -> Union[float, int, np.ndarray]:
    """
    Reduction Function returning the min element

    Parameters
    ----------
    items : list
        the items to reduce

    Returns
    -------
    float, int or :class:`numpy.ndarray`
        reduced items

    """
    return np.min(items)


def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    """
    Flattens a dictionary by concatenating all keys for subdicts with the
    current key separated by :param`sep`

    Parameters
    ----------
    d : dict
        the dictionary to flatten
    parent_key : str
        the key of the parent dict (ususally empty when called by user)
    sep : str
        the separator to separate the key from the subdict's key

    Returns
    -------
    dict
        the flattened dictionary

    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))

    return type(d)(items)


def unflatten_dict(dictionary: dict, sep: str = ".") -> dict:
    """
    Unflattens a dict, where keys and the keys from their subdirs are
    separated by :param:`sep`

    Parameters
    ----------
    dictionary : dict
        the dictionary to unflatten
    sep : str
        the separation string

    Returns
    -------

    """
    return_dict = {}
    for key, value in dictionary.items():
        parts = key.split(sep)
        d = return_dict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return return_dict


def reduce_dict(items: list, reduce_fn) -> dict:
    """
    A function to reduce all entries inside a dict

    Parameters
    ----------
    items : list
        a list of dicts to reduce
    reduce_fn : FunctionType
        a function to apply to all non-equal iterables

    Returns
    -------
    dict
        the reduced dict

    """

    result_dict = {}
    # assuming the type of all items is same for all queued logging dicts and
    # all dicts have the same keys

    flattened_dicts = [flatten_dict(_tmp, sep=".") for _tmp in items]

    # from list of dicts to dict of lists:
    for d in flattened_dicts:
        for k, v in d.items():
            try:
                result_dict[k].append(v)
            except KeyError:
                result_dict[k] = [v]

    for k, v in result_dict.items():
        # check if all items are equal
        equals = [_v == v[0] for _v in v[1:]]
        for idx, equality in enumerate(equals):
            if isinstance(equality, np.ndarray):
                equals[idx] = equality.all()
        if all(equals):
            # use first item since they are equal
            result_dict[k] = v[0]
        else:
            # apply reduce function
            result_dict[k] = reduce_fn(v)

    # unflatten reduced dict
    return unflatten_dict(result_dict, sep=".")


# string mapping for reduction functions
_REDUCTION_FUNCTIONS = {
    "last": reduce_last,
    "first": reduce_first,
    "mean": reduce_mean,
    "median": reduce_median,
    "max": reduce_max,
    "min": reduce_min
}


def possible_reductions() -> tuple:
    """
    Function returning a tuple containing all valid reduction strings

    Returns
    -------
    tuple
        a tuple containing all valid reduction strings
    """
    return tuple(_REDUCTION_FUNCTIONS.keys())


def get_reduction(reduce_type: str) -> Callable:
    """
    A getter function to get a specified reduction function by it's
    specifier string

    Parameters
    ----------
    reduce_type : str
        the reduction type

    Returns
    -------
    Callable
        the actual reduction function

    """
    return _REDUCTION_FUNCTIONS[reduce_type]
