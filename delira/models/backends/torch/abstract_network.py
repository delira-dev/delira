import abc
import torch
from delira.models.abstract_network import AbstractNetwork


class AbstractPyTorchNetwork(AbstractNetwork, torch.nn.Module):
    """
    Abstract Class for PyTorch Networks

    See Also
    --------
    `torch.nn.Module`
    :class:`AbstractNetwork`

    """
    @abc.abstractmethod
    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        **kwargs :
            keyword arguments (are passed to :class:`AbstractNetwork`'s `
            __init__ to register them as init kwargs

        """
        torch.nn.Module.__init__(self)
        AbstractNetwork.__init__(self, **kwargs)

    @abc.abstractmethod
    def forward(self, *inputs):
        """
        Forward inputs through module (defines module behavior)
        Parameters
        ----------
        inputs : list
            inputs of arbitrary type and number

        Returns
        -------
        Any
            result: module results of arbitrary type and number

        """
        raise NotImplementedError()

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
