import collections
import numpy as np


def recursively_convert_elements(element, check_type, conversion_fn):
    """
    Function to recursively convert all elements

    Parameters
    ----------
    element : Any
        the element to convert
    check_type : Any
        if ``element`` is of type ``check_type``, the conversion function will
        be applied to it
    conversion_fn : Any
        the function to apply to ``element`` if it is of type ``check_type``

    Returns
    -------
    Any
        the converted element

    """

    # convert element with conversion_fn
    if isinstance(element, check_type):
        return conversion_fn(element)

    # return string and arrays as is
    elif isinstance(element, (str, np.ndarray)):
        return element

    # recursively convert all keys and values of mapping and convert result
    # back to original mapping type
    # must be checked before iterable since most mappings are also a iterable
    elif isinstance(element, collections.Mapping):
        element = type(element)({
            recursively_convert_elements(k, check_type, conversion_fn):
                recursively_convert_elements(v, check_type, conversion_fn)
            for k, v in element.items()
        })
        return element

    # recursively convert all items of iterable and convert result back to
    # original iterable type
    elif isinstance(element, collections.Iterable):
        element = type(element)([recursively_convert_elements(x,
                                                              check_type,
                                                              conversion_fn)
                                 for x in element])
        return element

    # none of the previous cases is suitable for the element -> return as is
    return element


def _correct_zero_shape(arg):
    """
    Corrects the shape of numpy array to be at least 1d and returns the
    argument as is otherwise

    Parameters
    ----------
    arg : Any
        the argument which must be corrected in its shape if it's
        zero-dimensional

    Returns
    -------
    Any
        argument (shape corrected if necessary)
    """
    if arg.shape == ():
        arg = arg.reshape(1)

    return arg


def convert_to_numpy_identity(*args, **kwargs):
    """
    Corrects the shape of all zero-sized numpy arrays to be at least 1d

    Parameters
    ----------
    *args :
        positional arguments of potential arrays to be corrected
    **kwargs :
        keyword arguments of potential arrays to be corrected

    Returns
    -------

    """
    args = recursively_convert_elements(args, np.ndarray, _correct_zero_shape)

    kwargs = recursively_convert_elements(kwargs, np.ndarray,
                                          _correct_zero_shape)

    return args, kwargs
