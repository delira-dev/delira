import abc
import torch
from delira.models.abstract_network import AbstractNetwork


class AbstractTorchScriptNetwork(AbstractNetwork, torch.jit.ScriptModule):
    """
    Abstract Interface Class for TorchScript Networks. For more information
    have a look at https://pytorch.org/docs/stable/jit.html#torchscript

    Warnings
    --------
    In addition to the here defined API, a forward function must be
    implemented and decorated with ``@torch.jit.script_method``

    """

    @abc.abstractmethod
    def __init__(self, optimize=True, **kwargs):
        """

        Parameters
        ----------
        optimize : bool
            whether to optimize the network graph or not; default: True
        **kwargs :
            additional keyword arguments
            (passed to :class:`AbstractNetwork`)
        """
        torch.jit.ScriptModule.__init__(self, optimize=optimize)
        AbstractNetwork.__init__(self, **kwargs)

    def __call__(self, *args, **kwargs):
        """
        Calls Forward method

        Parameters
        ----------
        *args :
            positional arguments (passed to `forward`)
        **kwargs :
            keyword arguments (passed to `forward`)

        Returns
        -------
        Any
            result: module results of arbitrary type and number

        """
        return torch.jit.ScriptModule.__call__(self, *args, **kwargs)

    @staticmethod
    def prepare_batch(batch: dict, input_device, output_device):
        """
        Helper Function to prepare Network Inputs and Labels (convert them
        to correct type and shape and push them to correct devices)

        Parameters
        ----------
        batch : dict
            dictionary containing all the data
        input_device : torch.device
            device for network inputs
        output_device : torch.device
            device for network outputs

        Returns
        -------
        dict
            dictionary containing data in correct type and shape and on
            correct device

        """
        return_dict = {"data": torch.from_numpy(batch.pop("data")).to(
            input_device).to(torch.float)}

        for key, vals in batch.items():
            return_dict[key] = torch.from_numpy(vals).to(output_device).to(
                torch.float)

        return return_dict
