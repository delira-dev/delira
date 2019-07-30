import torch

from delira.models.backends.torch.abstract_network import \
    AbstractPyTorchNetwork


class DataParallelPyTorchNetwork(AbstractPyTorchNetwork,
                                 torch.nn.DataParallel):
    """
    A Wrapper around a :class:`AbstractPyTorchNetwork` instance to
    implement parallel training by splitting the batches
    """

    def __init__(self, module: AbstractPyTorchNetwork,
                 device_ids=None, output_device=None, dim=0):
        """

        Parameters
        ----------
        module : :class:`AbstractPyTorchNetwork`
            the module to wrap (will be replicated on all devices)
        device_ids : list
            a list containing the devices to use (either as strings or as
            :class:`chainer.backend.Device`).
        output_device : str or :class:`chainer.backend.Device`
            The output device
            Make sure, your labels are also on this device
            for loss calculation!
            If not specified, the second device of ``devices`` will be used
            for output gathering.
        dim : int
            the index of the batchdimension (usually 0, but can become
            e.g. 1 in NLP tasks)

        """

        AbstractPyTorchNetwork.__init__(self)
        torch.nn.DataParallel.__init__(self, module, device_ids, output_device,
                                       dim)

    def forward(self, *args, **kwargs):
        """
        Scatters the inputs (both positional and keyword arguments) across
        all devices, feeds them through model replicas and re-builds
        batches on output device

        Parameters
        ----------
        *args :
            positional arguments of arbitrary number and type
        **kwargs :
            keyword arguments of arbitrary number and type

        Returns
        -------
        Any
            combined output from all scattered models

        """
        return torch.nn.DataParallel.forward(*args, **kwargs)

    @property
    def closure(self):
        return self.module.closure

    @property
    def prepare_batch(self):
        return self.module.prepare_batch
