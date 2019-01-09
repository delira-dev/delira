import inspect
import logging
import os
import importlib
from collections import OrderedDict
from itertools import islice

logger = logging.getLogger(__name__)

try:

    import torch

    from torchvision import models as t_models
    from torch import nn
    from torch.nn import functional as F
    from torch import optim

    from ..models import AbstractPyTorchNetwork




    def save_checkpoint(file: str, model=None, optimizers={},
                        epoch=None, weights_only=False, **kwargs):
        """
        Save model's parameters

        Parameters
        ----------
        file : str
            filepath the model should be saved to
        model : AbstractNetwork or None
            the model which should be saved
            if None: empty dict will be saved as state dict
        optimizers : dict
            dictionary containing all optimizers
        epoch : int
            current epoch (will also be pickled)
        weights_only : bool
            whether or not to save only the model's weights or also save additional
            information (for easy loading)

        """
        if isinstance(model, torch.nn.DataParallel):
            _model = model.module
        else:
            _model = model

        if isinstance(_model, AbstractPyTorchNetwork):
            model_state = _model.state_dict()
        else:
            model_state = {}
            logger.debug("Saving checkpoint without Model")

        optim_state = OrderedDict()
        for key, val in optimizers.items():
            if isinstance(val, torch.optim.Optimizer):
                optim_state[key] = val.state_dict()

        if not optim_state:
            logger.debug("Saving checkpoint without Optimizer")

        if epoch is None:
            epoch = 0

        state = {"optimizer": optim_state,
                "model": model_state,
                "epoch": epoch}

        if not weights_only:

            source = inspect.getsource(_model.__class__)

            class_name_model = _model.__class__.__name__
            class_names_optim = OrderedDict()

            for key in optim_state.keys():
                class_names_optim[key] = optimizers[key].__class__.__name__

            parent_class = _model.__class__.__mro__[1].__name__

            init_kwargs = _model.init_kwargs

            torch.save({'source': source, 'cls_name_model': class_name_model,
                        'parent_class': parent_class, 'init_kwargs': init_kwargs,
                        'state_dict': state, 'cls_name_optim': class_names_optim},
                    file)

        else:
            torch.save(state, file)


    def load_checkpoint(file, weights_only=False, **kwargs):
        """
        Loads a saved model

        Parameters
        ----------
        file : str
            filepath to a file containing a saved model
        weights_only : bool
            whether the file contains only weights / only weights should be
            returned
        **kwargs:
            Additional keyword arguments (passed to torch.load)
            Especially "map_location" is important to change the device the
            state_dict should be loaded to

        Returns
        -------
        OrderedDict
            checkpoint state_dict if `weights_only=True`
        torch.nn.Module, OrderedDict, int
            Model, Optimizers, epoch with loaded state_dicts if `weights_only=False`

        """
        if weights_only:
            return torch.load(file, **kwargs)
        else:
            loaded_dict = torch.load(file, **kwargs)

            # import parent class
            exec("from ..models import " + loaded_dict["parent_class"])

            # execute pickled code (to get access to class)
            exec(loaded_dict["source"])

            # create class instance (default device: CPU)
            exec("model = " + loaded_dict["cls_name_model"] +
                "(**loaded_dict['init_kwargs'])")

            # check for "map_location" kwarg and use device of first weight tensor
            # as default argument (weight tensors should be all on same device)
            if loaded_dict["state_dict"]["model"]:
                default_device = next(
                    islice(
                        loaded_dict["state_dict"]["model"].values(), 1)
                ).device
            else:
                default_device = torch.device("cpu")

            map_location = kwargs.get("map_location",
                                    # use slicing instead of converting to list
                                    # to avoid memory overhead
                                    default_device)

            # push created class from CPU to suitable device
            locals()['model'].to(map_location)

            locals()['model'].load_state_dict(loaded_dict["state_dict"]["model"])

            optims = OrderedDict()

            for key in loaded_dict["cls_name_optim"].keys():
                exec("_optim = optim.%s(models.parameters())" %
                loaded_dict["cls_name_optim"][key])

                optims[key] = locals()['_optim']

            for key, val in optims.items():
                optims[key] = val.load_state_dict(
                    loaded_dict["state_dict"]["optimizer"][key])

            return locals()['model'], optims, loaded_dict["state_dict"]["epoch"]

except ImportError as e:
    raise e