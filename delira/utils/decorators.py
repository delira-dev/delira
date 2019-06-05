import warnings
from functools import wraps

import numpy as np

from delira import get_backends


def dtype_func(class_object):
    """
    Decorator to Check whether the first argument of the decorated function is
    of a certain type

    Parameters
    ----------
    class_object : Any
        type the first function argument should have

    Returns
    -------
    Wrapped Function

    Raises
    ------
    AssertionError
        First argument of decorated function is not of given type

    """
    def instance_checker(func):
        @wraps(func)
        def func_wrapper(checked_object, *args, **kwargs):
            assertion_str = "Argument 1 is not of type %s but of type %s" % \
                            (class_object.__name__,
                             checked_object.__class__.__name__)

            assert isinstance(checked_object, class_object), assertion_str
            return func(checked_object, *args, **kwargs)
        return func_wrapper
    return instance_checker


def classtype_func(class_object):
    """
    Decorator to Check whether the first argument of the decorated function is
    a subclass of a certain type

    Parameters
    ----------
    class_object : Any
        type the first function argument should be subclassed from

    Returns
    -------
    Wrapped Function

    Raises
    ------
    AssertionError
        First argument of decorated function is not a subclass of given type

    """
    def subclass_checker(func):
        @wraps(func)
        def func_wrapper(checked_object, *args, **kwargs):
            assertion_str = "Argument 1 is not subclass of %s but of type %s" \
                            % (class_object.__name__, checked_object.__name__)

            assert issubclass(checked_object, class_object), assertion_str
            return func(checked_object, *args, **kwargs)
        return func_wrapper
    return subclass_checker


def make_deprecated(new_func):
    """
    Decorator which raises a DeprecationWarning for the decorated object

    Parameters
    ----------
    new_func : Any
        new function which should be used instead of the decorated one

    Returns
    -------
    Wrapped Function

    Raises
    ------
    Deprecation Warning

    """
    def deprecation(func):
        @wraps(func)
        def func_wrapper(*args, **kwargs):
            if not isinstance(new_func, str):
                new_func_name = new_func.__name__
            else:
                new_func_name = new_func

            if func.__name__ == '__init__':
                old_func_name = func.__class__.__name__
            else:
                old_func_name = func.__name__
            warnings.warn(DeprecationWarning("%s is deprecated in favor of %s"
                                             " and will be removed at next "
                                             "release" % (old_func_name,
                                                          new_func_name)))
            return func(*args, **kwargs)

        return func_wrapper
    return deprecation


numpy_array_func = dtype_func(np.ndarray)


if "TORCH" in get_backends():
    import torch
    torch_tensor_func = dtype_func(torch.Tensor)
    torch_module_func = dtype_func(torch.nn.Module)
