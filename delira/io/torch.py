import inspect
import logging
import os
import importlib
from collections import OrderedDict
from itertools import islice
from delira import get_backends

logger = logging.getLogger(__name__)

if "TORCH" in get_backends():

    import torch
    from ..models import AbstractPyTorchNetwork

    def save_checkpoint(file: str, model=None, optimizers={},
                        epoch=None, **kwargs):
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

        torch.save(state, file, **kwargs)

    def load_checkpoint(file, **kwargs):
        """
        Loads a saved model

        Parameters
        ----------
        file : str
            filepath to a file containing a saved model
        **kwargs:
            Additional keyword arguments (passed to torch.load)
            Especially "map_location" is important to change the device the
            state_dict should be loaded to

        Returns
        -------
        OrderedDict
            checkpoint state_dict

        """
        checkpoint = torch.load(file, **kwargs)

        if not all([_key in checkpoint
                    for _key in ["model", "optimizer", "epoch"]]):
            return checkpoint['state_dict']
        return checkpoint
