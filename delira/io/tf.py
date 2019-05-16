import logging
import tensorflow as tf
logger = logging.getLogger(__name__)

import typing

from ..models import AbstractTfEagerNetwork


def save_checkpoint(file: str, model=None):
    """
    Save model's parameters contained in it's graph

    Parameters
    ----------
    file : str
        filepath the model should be saved to
    model : TfNetwork
        the model which should be saved
    """
    tf.train.Saver().save(model._sess, file)


def load_checkpoint(file: str, model=None):
    """
    Loads a saved model

    Parameters
    ----------
    file : str
        filepath to a file containing a saved model
    model : TfNetwork
        the model which should be loaded
    """

    # following operation adds AssignVariableOps to the graph, keep an eye on this for memory leak
    tf.train.Saver().restore(model._sess, file)
    return {}


def _create_varlist(model: AbstractTfEagerNetwork = None,
                    optimizer: typing.Dict[str, tf.train.Optimizer] = None):
    variable_list = []

    if model is not None:
        variable_list += model.variables

    if optimizer is not None:
        for k, v in optimizer.items():
            variable_list += v.variables()

    return variable_list


def save_checkpoint_eager(file, model: AbstractTfEagerNetwork = None,
                          optimizer: typing.Dict[str, tf.train.Optimizer] = None,
                          epoch=None):
    variable_list = _create_varlist(model, optimizer)
    saver = tf.contrib.eager.Saver(variable_list)
    saver.save(file, global_step=epoch)


def load_checkpoint_eager(file, model: AbstractTfEagerNetwork = None,
                          optimizer: typing.Dict[str, tf.train.Optimizer] = None
                          ):

    variable_list = _create_varlist(model, optimizer)

    saver = tf.contrib.eager.Saver(variable_list)
    saver.restore(file)

    return {"model": model, "optimizer": optimizer}