import abc
import tensorflow as tf
import numpy as np
from delira.models.abstract_network import AbstractNetwork


class AbstractTfEagerNetwork(AbstractNetwork, tf.keras.layers.Layer):
    """
    Abstract Network for TF eager execution backend.
    All models to use with this backend should be derived from this class
    """
    def __init__(self, data_format="channels_first", trainable=True,
                 name=None, dtype=None, **kwargs):
        """

        Parameters
        ----------
        data_format : str
            the accepted data format (default: 'channels_first')
        trainable : wheter or not the model is trainable (default: True)
        name : str
            the network's name
        dtype :
            the dtype to use for the model's parameters
        **kwargs :
            additional keyword arguments (will be registered as
            ``init_kwargs``)

        """
        tf.keras.layers.Layer.__init__(self, trainable=trainable,
                                       name=name, dtype=dtype)

        AbstractNetwork.__init__(self, **kwargs)

        self.data_format = data_format
        self.device = "/cpu:0"

    @abc.abstractmethod
    def call(self, *args, **kwargs):
        """
        Defines the model's forward pass

        Parameters
        ----------
        *args :
            arbitrary positional arguments
        **kwargs :
            arbbitrary keyword arguments

        Raises
        ------
        NotImplementedError
            If not overwritten by subclass

        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """
        Executes the modules forward pass

        Parameters
        ----------
        *args :
            arbitrary positional arguments
        **kwargs :
            arbbitrary keyword arguments

        """

        return self.call(*args, **kwargs)

    @staticmethod
    def prepare_batch(batch: dict, input_device, output_device):
        new_batch = {}
        with tf.device(output_device):
            new_batch["label"] = tf.convert_to_tensor(
                batch["label"].astype(np.float32))

        with tf.device(input_device):
            for k, v in batch.items():
                if k == "label":
                    continue
                new_batch[k] = tf.convert_to_tensor(v.astype(np.float32))

        return new_batch
