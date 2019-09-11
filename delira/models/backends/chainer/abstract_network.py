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

    @staticmethod
    def closure(model, data_dict: dict, optimizers: dict, losses: dict,
                iter_num, fold=0, **kwargs):
        """
        default closure method to do a single training step;
        Could be overwritten for more advanced models

        Parameters
        ----------
        model : :class:`AbstractChainerNetwork`
            trainable model
        data_dict : dict
            dictionary containing the data
        optimizers : dict
            dictionary of optimizers to optimize model's parameters;
            ignored here, just passed for compatibility reasons
        losses : dict
            dict holding the losses to calculate errors;
            ignored here, just passed for compatibility reasons
        iter_num: int
            the number of of the current iteration in the current epoch;
            Will be restarted at zero at the beginning of every epoch
        fold : int
            Current Fold in Crossvalidation (default: 0)
        **kwargs:
            additional keyword arguments

        Returns
        -------
        dict
            Loss values (with same keys as input dict losses; will always
            be empty here)
        dict
            dictionary containing all predictions

        """
        assert (optimizers and losses) or not optimizers, \
            "Criterion dict cannot be emtpy, if optimizers are passed"

        loss_vals = {}
        total_loss = 0

        inputs = data_dict["data"]
        preds = model(inputs)

        for key, crit_fn in losses.items():
            _loss_val = crit_fn(preds["pred"], data_dict["label"])
            loss_vals[key] = _loss_val.item()
            total_loss += _loss_val

        model.cleargrads()
        total_loss.backward()
        optimizers['default'].update()
        for k, v in preds.items():
            v.unchain()
        return loss_vals, preds
