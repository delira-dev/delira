import tensorflow as tf
from delira.models.backends.tf_eager.abstract_network import \
    AbstractTfEagerNetwork


class DataParallelTfEagerNetwork(AbstractTfEagerNetwork):
    """
    DataParallel Module for the TF eager execution backend

    Warnings
    --------
    This Module is highly experimental and not guaranteed to work properly!
    """

    def __init__(self, module, devices):
        """

        Parameters
        ----------
        module : :class:`AbstractTfEagerNetwork`
            the module to scatter across different devices
        devices : list
            list of ints specifying the GPU indices
        """
        super().__init__()

        self._closure = module.closure
        self._prepare_batch = module.pepare_batch

        self.module = tf.keras.utils.multi_gpu_model(module, devices, True)

    def call(self, *args, **kwargs):
        """
        Defines the forward pass of the module

        Parameters
        ----------
        *args :
            arbitrary positional arguments
        **kwargs :
            arbitrary keyword arguments

        """
        return self.module.call(*args, **kwargs)

    @property
    def closure(self):
        return self._closure

    @property
    def prepare_batch(self):
        return self._prepare_batch
