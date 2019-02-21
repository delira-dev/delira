import logging
file_logger = logging.getLogger(__name__)

from delira import get_backends

if "TORCH" in get_backends():
    import torch
    from torchvision import models as t_models
    from delira.models.abstract_network import AbstractPyTorchNetwork

    class ClassificationNetworkBasePyTorch(AbstractPyTorchNetwork):
        """
        Implements basic classification with ResNet18

        References
        ----------
        https://arxiv.org/abs/1512.03385

        See Also
        --------
        :class:`AbstractPyTorchNetwork`

        """

        def __init__(self, in_channels: int, n_outputs: int, **kwargs):
            """

            Parameters
            ----------
            in_channels : int
                number of input_channels
            n_outputs : int
                number of outputs (usually same as number of classes)

            """
            # register params by passing them as kwargs to parent class __init__
            super().__init__(in_channels=in_channels,
                             n_outputs=n_outputs,
                             **kwargs)

            self.module = self._build_model(in_channels, n_outputs, **kwargs)

            for key, value in kwargs.items():
                setattr(self, key, value)

        def forward(self, input_batch: torch.Tensor):
            """
            Forward input_batch through network

            Parameters
            ----------
            input_batch : torch.Tensor
                batch to forward through network

            Returns
            -------
            torch.Tensor
                Classification Result

            """

            return self.module(input_batch)

        @staticmethod
        def closure(model: AbstractPyTorchNetwork, data_dict: dict,
                    optimizers: dict, criterions={}, metrics={},
                    fold=0, **kwargs):
            """
            closure method to do a single backpropagation step


            Parameters
            ----------
            model : :class:`ClassificationNetworkBasePyTorch`
                trainable model
            data_dict : dict
                dictionary containing the data
            optimizers : dict
                dictionary of optimizers to optimize model's parameters
            criterions : dict
                dict holding the criterions to calculate errors
                (gradients from different criterions will be accumulated)
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
                Loss values (with same keys as input dict criterions)
            list
                Arbitrary number of predictions as torch.Tensor

            Raises
            ------
            AssertionError
                if optimizers or criterions are empty or the optimizers are not
                specified

            """

            assert (optimizers and criterions) or not optimizers, \
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

                inputs = data_dict.pop("data")
                preds = model(inputs)

                if data_dict:

                    for key, crit_fn in criterions.items():
                        _loss_val = crit_fn(preds, *data_dict.values())
                        loss_vals[key] = _loss_val.detach()
                        total_loss += _loss_val

                    with torch.no_grad():
                        for key, metric_fn in metrics.items():
                            metric_vals[key] = metric_fn(
                                preds, *data_dict.values())

            if optimizers:
                optimizers['default'].zero_grad()
                # perform loss scaling via apex if half precision is enabled
                with optimizers["default"].scale_loss(total_loss) as scaled_loss:
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

            for key, val in {**metric_vals, **loss_vals}.items():
                logging.info({"value": {"value": val.item(), "name": key,
                                        "env_appendix": "_%02d" % fold
                                        }})

            logging.info({'image_grid': {"images": inputs, "name": "input_images",
                                         "env_appendix": "_%02d" % fold}})

            return metric_vals, loss_vals, [preds]

        @staticmethod
        def _build_model(in_channels: int, n_outputs: int, **kwargs):
            """
            builds actual model (resnet 18)

            Parameters
            ----------
            in_channels : int
                number of input channels
            n_outputs : int
                number of outputs (usually same as number of classes)
            **kwargs : dict
                additional keyword arguments

            Returns
            -------
            torch.nn.Module
                created model

            """

            _model = t_models.resnet18(
                pretrained=False, num_classes=n_outputs, **kwargs)
            _model.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7,
                                           stride=2, padding=3, bias=False)

            return _model

        @staticmethod
        def prepare_batch(batch: dict, input_device, output_device):
            """
            Helper Function to prepare Network Inputs and Labels (convert them to
            correct type and shape and push them to correct devices)

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
                dictionary containing data in correct type and shape and on correct
                device

            """
            return_dict = {"data": torch.from_numpy(batch.pop("data")).to(
                input_device).to(torch.float)}

            for key, vals in batch.items():
                return_dict[key] = torch.from_numpy(vals).to(
                    output_device).squeeze(-1).to(torch.long)

            return return_dict
