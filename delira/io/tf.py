import logging
import tensorflow as tf
logger = logging.getLogger(__name__)


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
