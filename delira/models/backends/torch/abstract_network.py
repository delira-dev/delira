import abc
import torch
from delira.models.abstract_network import AbstractNetwork

from .utils import scale_loss


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

    @staticmethod
    def closure(model, data_dict: dict, optimizers: dict, losses={},
                metrics={}, fold=0, **kwargs):
        """
        closure method to do a single backpropagation step

        Parameters
        ----------
        model : :class:`AbstractPyTorchNetwork`
            trainable model
        data_dict : dict
            dictionary containing the data
        optimizers : dict
            dictionary of optimizers to optimize model's parameters
        losses : dict
            dict holding the losses to calculate errors
            (gradients from different losses will be accumulated)
        metrics : dict
            dict holding the metrics to calculate
        fold : int
            Current Fold in Crossvalidation (default: 0)
        **kwargs:
            additional keyword arguments

        Returns
        -------
        dict
            Metric values (with same keys as input dict metrics)
        dict
            Loss values (with same keys as input dict losses)
        list
            Arbitrary number of predictions as torch.Tensor

        Raises
        ------
        AssertionError
            if optimizers or losses are empty or the optimizers are not
            specified
        """

        assert (optimizers and losses) or not optimizers, \
            "Criterion dict cannot be emtpy, if optimizers are passed"

        loss_vals = {}
        metric_vals = {}
        total_loss = 0

        # choose suitable context manager:
        if optimizers:
            context_man = torch.enable_grad

        else:
            context_man = torch.no_grad

        with context_man():

            # predict
            inputs = data_dict.pop("data")
            preds = model(inputs)

            # calculate losses
            for key, crit_fn in losses.items():
                _loss_val = crit_fn(preds["pred"], data_dict["label"])
                loss_vals[key] = _loss_val.item()
                total_loss += _loss_val

            # calculate metrics
            with torch.no_grad():
                for key, metric_fn in metrics.items():
                    metric_vals[key] = metric_fn(
                        preds["pred"], data_dict["label"]).item()

        if optimizers:
            optimizers['default'].zero_grad()
            # perform loss scaling via apex if half precision is enabled
            with scale_loss(total_loss, optimizers["default"]) as scaled_loss:
                scaled_loss.backward()
            optimizers['default'].step()

        else:

            # add prefix "val" in validation mode
            eval_loss_vals, eval_metrics_vals = {}, {}
            for key in loss_vals.keys():
                eval_loss_vals["val_" + str(key)] = loss_vals[key]

            for key in metric_vals:
                eval_metrics_vals["val_" + str(key)] = metric_vals[key]

            loss_vals = eval_loss_vals
            metric_vals = eval_metrics_vals

        return metric_vals, loss_vals, {k: v.detach()
                                        for k, v in preds.items()}
