import collections
import numpy as np
from tqdm import tqdm


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


def create_iterator(base_iterable: collections.Iterable,
                    verbose: bool = False, unit: str = "it",
                    total_num: int = None, desc: str = None,
                    enum: bool = False, **kwargs):
    """
    Function to wrap an iterable to provide verbosity and enumeration if
    desired

    Parameters
    ----------
    base_iterable : Iterable
        the iterable to wrap
    verbose : bool
        whether to add verbosity; defaults to False
    unit : str
        the unit to show; Will only be used if :param:`verbose` is True;
        defaults to None
    total_num : int
        the maximum number of samples in the iterator;
        necessary in case the iterable does not have a length attribute;
        Will only be used if :param:`verbose` is True; defaults to None
    desc : str
        description of the current iterable;
        Will only be used if :param:`verbose` is True;
    enum : bool
        whether to enumerate over the iterable
    **kwargs :
        arbitrary keyword arguments passed to :class:`tqdm.tqdm`

    Returns
    -------
    Iterable
        a wrapped iterable with the desired options but the same content

    See Also
    --------
    :class:`tqdm.tqdm` for accepted keyword arguments

    """

    if not unit.startswith(" "):
        unit = " " + unit

    if verbose:
        iterable = tqdm(base_iterable, unit=unit,
                        total=total_num, desc=desc, **kwargs)

    else:
        iterable = base_iterable

    if enum:
        iterable = enumerate(iterable)

    return iterable
