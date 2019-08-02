import chainer
from delira.models.backends.chainer import DataParallelChainerOptimizer
from delira.training.utils import convert_to_numpy_identity, \
    recursively_convert_elements


def _single_element_tensor_conversion(element):
    element.to_cpu()
    return element.array


def convert_to_numpy(*args, **kwargs):
    """
    Converts all chainer variables in args and kwargs to numpy array

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
    args = recursively_convert_elements(args, chainer.Variable,
                                        _single_element_tensor_conversion)

    kwargs = recursively_convert_elements(kwargs, chainer.Variable,
                                          _single_element_tensor_conversion)

    return convert_to_numpy_identity(*args, **kwargs)


def create_optims_default(model, optim_cls, **optimizer_params):
    """
    Default function to create a single optimizer for chainer
    (also supports Data-Parallel)

    Parameters
    ----------
    model : :class:`chainer.Link`
        the model, which should be updated by the optimizer
    optim_cls : type
        the optimizer class implementing the actual parameter update
    optimizer_params : dict
        the params used for initializing an instance of ``optim_cls``

    Returns
    -------
    dict
        dictionary containing the created optimizer (key: "default")

    """
    if issubclass(optim_cls, DataParallelChainerOptimizer):
        optim = optim_cls.from_optimizer_class(**optimizer_params)

    else:
        optim = optim_cls(**optimizer_params)

    optim = optim.setup(model)

    return {"default": optim}
