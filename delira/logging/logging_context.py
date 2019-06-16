import typing

from delira.logging.utils import logger_exists, register_logger, \
    unregister_logger, log as _log
from delira.logging.base_logger import make_logger


class LoggingContext(object):

    def __init__(
            self,
            name,
            initialize_if_missing=False,
            destroy_on_exit=False,
            **kwargs):
        if logger_exists(name):
            self._name = name
        elif initialize_if_missing:
            register_logger(make_logger(**kwargs), name)
            self._name = name
            destroy_on_exit = True
        else:
            raise ValueError("No valid logger for name %s and "
                             "'initialize_if_missing' is False" % name)

        self._destroy_on_exit = destroy_on_exit

    def __enter__(self):
        global log
        log = self.log
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._destroy_on_exit:
            _logger = unregister_logger(self._name)
            del _logger

        global log
        log = _log

    def log(self, msg: typing.Union[dict, list, str]):
        _log(msg, self._name)
