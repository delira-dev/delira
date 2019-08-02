import torch

from delira.utils.decorators import dtype_func
from delira.training.utils import convert_to_numpy_identity
from delira.training.utils import recursively_convert_elements


@dtype_func(torch.nn.Module)
def create_optims_default(model, optim_cls, **optim_params):
    """
    Function to create a optimizer dictionary
    (in this case only one optimizer for the whole network)

    Parameters
    ----------
    model : :class:`AbstractPyTorchNetwork`
        model whose parameters should be updated by the optimizer
    optim_cls :
        Class implementing an optimization algorithm
    **optim_params :
        Additional keyword arguments (passed to the optimizer class

    Returns
    -------
    dict
        dictionary containing all created optimizers
    """
    return {"default": optim_cls(model.parameters(), **optim_params)}


def _single_element_tensor_conversion(element):
    return element.cpu().detach().numpy()


def convert_to_numpy(*args, **kwargs):
    """
    Converts all :class:`torch.Tensor` in args and kwargs to numpy array

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
    args = recursively_convert_elements(args, torch.Tensor,
                                        _single_element_tensor_conversion)

    kwargs = recursively_convert_elements(kwargs, torch.Tensor,
                                          _single_element_tensor_conversion)

    return convert_to_numpy_identity(*args, **kwargs)
