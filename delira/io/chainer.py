from delira import get_backends

import chainer
import numpy as np


def save_checkpoint(file, model=None, optimizers=None, epoch=None):
    """
    Saves the given checkpoint

    Parameters
    ----------
    file : str
        string containing the path, the state should be saved to
    model : :class:`AbstractChainerNetwork`
    optimizers : dict
        dictionary containing all optimizers
    epoch : int
        the current epoch

    """
    save_state = {}

    # check if there is a model to save
    if model is not None:
        save_state["model"] = model

    # check if there are any optimizers to save
    if optimizers is not None and optimizers:
        save_state["optimizers"] = optimizers

    if epoch is not None:
        save_state["epoch"] = None

    # check if save_state is not empty (saving an empty state is useless)
    if save_state:
        chainer.serializers.save_npz(file, save_state)


def load_checkpoint(file):
    """
    Loads a state from a given file

    Parameters
    ----------
    file : str
        string containing the path to the file containing the saved state

    Returns
    -------
    dict
        the loaded state

    """
    state_file = np.load(file, allow_pickle=True)

    state = {"model": None, "optimizers": None, "epoch": None}

    state.update({k: v.item() for k, v in state_file.items()})

    return state
