import logging
import joblib
logger = logging.getLogger(__name__)


def save_checkpoint(file: str, model=None, epoch=None, **kwargs):
    """
    Save model's parameters

    Parameters
    ----------
    file : str
        filepath the model should be saved to
    model : AbstractNetwork or None
        the model which should be saved
        if None: empty dict will be saved as state dict
    epoch : int
        current epoch (will also be pickled)

    """

    return_val = joblib.dump({"model": model, "epoch": epoch}, file, **kwargs)
    return return_val


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
    return joblib.load(file, **kwargs)
