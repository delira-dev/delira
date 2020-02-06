from delira.logging.base_logger import Logger
from collections import OrderedDict

# Registry dict containing all registered available Loggers
# Use Ordered Dict here to use first logger for logging if no name was given
_AVAILABLE_LOGGERS = OrderedDict()


def log(msg: dict, name=None):
    """
    Global logging function

    Parameters
    ----------
    msg : dict
        the message to log; Should be a dict, where the keys indicate the
        logging function to execute, and the corresponding value holds
        the arguments necessary to execute this function
    name : str
        the name of the logger to use;
        if None: the last logger will be used

    Raises
    ------
    AssertionError
        if the logger with the specified name does not exist
    AssertionError
        if the returned object is not a logger

    Returns
    -------
    Any
        the value obtained by the loggers ``log`` function

    """

    # use last name if no name is present
    if name is None:
        name = get_available_loggers()[-1]

    assert logger_exists(name)
    _logger = get_logger(name)

    assert isinstance(_logger, Logger)

    return _logger.log(msg)


def logger_exists(name: str):
    """
    Check if logger exists

    Parameters
    ----------
    name : str
        the name to check the existence for

    Returns
    -------
    bool
        whether a logger with the given name exists

    """
    return name in _AVAILABLE_LOGGERS


def register_logger(logger: Logger, name: str, overwrite=False):
    """
    Register a new logger to the Registry

    Parameters
    ----------
    logger : :class:`delira.logging.base_logger.Logger`
        the logger to register
    name : str
        the corresponding name, to register the logger at
    overwrite : bool
        whether or not to overwrite existing loggers if necessary

    Returns
    -------
    :class:`delira.logging.base_logger.Logger`
        the registered logger object

    """

    if not logger_exists(name) or overwrite:
        _AVAILABLE_LOGGERS[name] = logger

    return get_logger(name)


def unregister_logger(name: str):
    """
    Unregisters a logger from the registry

    Parameters
    ----------
    name : str
        the name of the logger to unregister

    Returns
    -------
    :class:`delira.logging.base_logger.Logger`
        the registered logger object
    """
    return _AVAILABLE_LOGGERS.pop(name)


def get_logger(name):
    """
    Returns a logger from the registry

    Parameters
    ----------
    name : str
        the name indicating the logger to return

    Returns
    -------
    :class:`delira.logging.base_logger.Logger`
        the specified logger object

    """
    return _AVAILABLE_LOGGERS[name]


def get_available_loggers():
    """
    Gets names for all registered loggers

    Returns
    -------
    tuple
        a tuple of strings specifying the names of all registered loggers

    """
    return tuple(_AVAILABLE_LOGGERS.keys())
