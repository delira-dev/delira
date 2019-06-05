import tensorflow as tf
from tensorflow.python.eager.context import context, EAGER_MODE, GRAPH_MODE

from delira.training.utils import convert_to_numpy_identity, \
    recursively_convert_elements


def _single_element_tensor_conversion(element):
    return element.numpy()


def convert_to_numpy(*args, **kwargs):
    """
    Converts all tf tensors in args and kwargs to numpy array

    Parameters
    ----------
    *args :
        positional arguments of arbitrary number and type
    **kwargs :
        keyword arguments of arbitrary number and type

    Returns
    -------
    list
        converted positional arguments
    dict
        converted keyboard arguments
    """
    args = recursively_convert_elements(args, tf.Tensor,
                                        _single_element_tensor_conversion)

    kwargs = recursively_convert_elements(kwargs, tf.Tensor,
                                          _single_element_tensor_conversion)

    return convert_to_numpy_identity(args, kwargs)


def create_optims_default(optim_cls, **optim_params):
    """
    Function to create a optimizer dictionary
    (in this case only one optimizer)

    Parameters
    ----------
    optim_cls :
        Class implementing an optimization algorithm
    **optim_params :
        Additional keyword arguments (passed to the optimizer class)

    Returns
    -------
    dict
        dictionary containing all created optimizers
    """
    return {"default": optim_cls(**optim_params)}


# hacky switch function
def switch_tf_execution_mode(mode: str):
    """
    Function to change Tensorflow execution mode

    Parameters
    ----------
    mode : str
        the new execution mode, must be of one of
        ['eager', 'eager_mode', 'graph', 'graph_mode']

    Raises
    ------
    ValueError
        If the given mode is not any of the above mentioned

    """
    mode = mode.lower()

    mode = mode.replace("_mode", "")

    if mode == "eager":
        _mode = EAGER_MODE
    elif mode == "graph":
        _mode = GRAPH_MODE
    else:
        raise ValueError("Invalid Execution mode given: %s" % mode)

    ctx = context()._eager_context
    ctx.mode = _mode
    ctx.is_eager = _mode == EAGER_MODE
