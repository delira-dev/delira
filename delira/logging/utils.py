import typing
from delira.logging.base_logger import Logger


_AVAILABLE_LOGGERS = {}


def log(msg: typing.Union[dict, list, str], name=None):

    if name is None:
        _logger = list(_AVAILABLE_LOGGERS.values())[0]
    else:
        _logger = _AVAILABLE_LOGGERS[name]

    assert isinstance(_logger, Logger)

    return _logger.log(msg)


def logger_exists(name: str):
    return name in _AVAILABLE_LOGGERS


def register_logger(logger: Logger, name: str, overwrite=False):

    if name not in _AVAILABLE_LOGGERS or overwrite:
        _AVAILABLE_LOGGERS[name] = logger

    return _AVAILABLE_LOGGERS[name]


def unregister_logger(name: str):
    return _AVAILABLE_LOGGERS.pop(name)


def get_logger(name):
    return _AVAILABLE_LOGGERS[name]


def get_available_loggers():
    return list(_AVAILABLE_LOGGERS.keys())