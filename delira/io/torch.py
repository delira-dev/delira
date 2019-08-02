from delira.models.backends.torchscript import AbstractTorchScriptNetwork
from delira.models.backends.torch import AbstractPyTorchNetwork
import torch
import logging
import os
from collections import OrderedDict

logger = logging.getLogger(__name__)


def save_checkpoint_torch(file: str, model=None, optimizers=None,
                          epoch=None, **kwargs):
    """
    Save checkpoint

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
    if optimizers is None:
        optimizers = {}
    if isinstance(model, torch.nn.DataParallel):
        _model = model.module
    else:
        _model = model

    if isinstance(_model, (AbstractPyTorchNetwork,
                           AbstractTorchScriptNetwork)):
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


def load_checkpoint_torch(file, **kwargs):
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


def save_checkpoint_torchscript(file: str, model=None, optimizers=None,
                                epoch=None, **kwargs):
    """
    Save current checkpoint to two different files:
        1.) ``file + "_model.ptj"``: Will include the state of the model
            (including the graph; this is the opposite to
            :func:`save_checkpoint`)
        2.) ``file + "_trainer_state.pt"``: Will include the states of all
            optimizers and the current epoch (if given)

    Parameters
    ----------
    file : str
        filepath the model should be saved to
    model : AbstractPyTorchJITNetwork or None
        the model which should be saved
        if None: empty dict will be saved as state dict
    optimizers : dict
        dictionary containing all optimizers
    epoch : int
        current epoch (will also be pickled)

    """

    # remove file extension if given
    if optimizers is None:
        optimizers = {}
    if any([file.endswith(ext) for ext in [".pth", ".pt", ".ptj"]]):

        file, old_ext = file.rsplit(".", 1)

        if old_ext != "ptj":
            logger.info("File extension was changed from %s to ptj to "
                        "indicate that the current module is a "
                        "torchscript module (including the graph)")

    if isinstance(model, AbstractTorchScriptNetwork):
        torch.jit.save(model, file + ".model.ptj")

    if optimizers or epoch is not None:
        save_checkpoint_torch(file + ".trainer_state.pt", None,
                              optimizers=optimizers, epoch=epoch, **kwargs)


def load_checkpoint_torchscript(file: str, **kwargs):
    """
    Loads a saved checkpoint consisting of 2 files
    (see :func:`save_checkpoint_jit` for details)

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

    # load model
    if os.path.isfile(file):
        model_file = file
    elif os.path.isfile(file.replace(".ptj", ".model.ptj")):
        model_file = file.replace(".ptj", ".model.ptj")
    else:
        raise ValueError("No Model File found for %s" % file)

    # load trainer state (if possible)
    trainer_file = model_file.replace(".model.ptj", ".trainer_state.pt")
    if os.path.isfile(trainer_file):
        trainer_state = load_checkpoint_torch(trainer_file, **kwargs)

    else:
        trainer_state = {"optimizer": {},
                         "epoch": None}

    trainer_state.update({"model": torch.jit.load(model_file)})

    return trainer_state
