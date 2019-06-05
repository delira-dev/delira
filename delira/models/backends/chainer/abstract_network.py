import abc
import chainer
import numpy as np

from delira.models.abstract_network import AbstractNetwork


# Use this Mixin Class to set __call__ to None, because there is an
# internal check inside chainer.Link.__call__ for other __call__ methods
# of parent classes to be not None. If this would be the case,
# this function would be executed instead of our forward
class ChainerMixin(AbstractNetwork):
    __call__ = None


class AbstractChainerNetwork(chainer.Chain, ChainerMixin):
    """
    Abstract Class for Chainer Networks
    """

    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        **kwargs :
            keyword arguments of arbitrary number and type
            (will be registered as ``init_kwargs``)

        """
        chainer.Chain.__init__(self)
        AbstractNetwork.__init__(self, **kwargs)

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> dict:
        """
        Feeds Arguments through the network

        Parameters
        ----------
        *args :
            positional arguments of arbitrary number and type
        **kwargs :
            keyword arguments of arbitrary number and type

        Returns
        -------
        dict
            dictionary containing all computation results

        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> dict:
        """
        Makes instances of this class callable.
        Calls the ``forward`` method.

        Parameters
        ----------
        *args :
            positional arguments of arbitrary number and type
        **kwargs :
            keyword arguments of arbitrary number and type

        Returns
        -------
        dict
            dictionary containing all computation results

        """

        return chainer.Chain.__call__(self, *args, **kwargs)

    @staticmethod
    def prepare_batch(batch: dict, input_device, output_device):
        """
        Helper Function to prepare Network Inputs and Labels (convert them
        to correct type and shape and push them to correct devices)

        Parameters
        ----------
        batch : dict
            dictionary containing all the data
        input_device : chainer.backend.Device or string
            device for network inputs
        output_device : torch.device
            device for network outputs

        Returns
        -------
        dict
            dictionary containing data in correct type and shape and on
            correct device

        """
        new_batch = {k: chainer.as_variable(v.astype(np.float32))
                     for k, v in batch.items()}

        for k, v in new_batch.items():
            if k == "data":
                device = input_device
            else:
                device = output_device

            # makes modification inplace!
            v.to_device(device)

        return new_batch
