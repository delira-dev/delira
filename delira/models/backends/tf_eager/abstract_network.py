import abc
import typing
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
            arbitrary keyword arguments

        """

        return self.call(*args, **kwargs)

    @staticmethod
    def prepare_batch(batch: dict, input_device, output_device):
        """
        Helper Function to prepare Network Inputs and Labels (convert them to
        correct type and shape and push them to correct devices)

        Parameters
        ----------
        batch : dict
            dictionary containing all the data
        input_device : str
            device for module inputs
        output_device : str
            device for module outputs

        Returns
        -------
        dict
            dictionary containing data in correct type and shape and on correct
            device

        """
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

    @staticmethod
    def closure(model, data_dict: dict,
                optimizers: typing.Dict[str, tf.train.Optimizer], losses={},
                metrics={}, fold=0, **kwargs):

        loss_vals, metric_vals = {}, {}

        # calculate loss with graph created by gradient taping
        with tf.GradientTape() as tape:
            preds = model(data_dict["data"])
            total_loss = None
            for k, loss_fn in losses.items():
                _loss_val = loss_fn(preds["pred"],
                                    data_dict["label"])
                loss_vals[k] = _loss_val.numpy()
                if total_loss is None:
                    total_loss = _loss_val
                else:
                    total_loss += _loss_val

        # calculate gradients
        grads = tape.gradient(total_loss,
                              model.trainable_variables)

        for k, metric_fn in metrics.items():
            metric_vals[k] = metric_fn(
                preds["pred"],
                data_dict["label"]).numpy()

        if optimizers:
            # perform optimization step
            optimizers["default"].apply_gradients(
                zip(grads, model.trainable_variables))
        else:
            # add prefix "val" in validation mode
            eval_losses, eval_metrics = {}, {}
            for key in loss_vals.keys():
                eval_losses["val_" + str(key)] = loss_vals[key]

            for key in metric_vals:
                eval_metrics["val_" + str(key)] = metric_vals[key]

            loss_vals = eval_losses
            metric_vals = eval_metrics

        return metric_vals, loss_vals, preds
