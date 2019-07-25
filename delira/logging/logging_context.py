from delira.logging.registry import logger_exists, register_logger, \
    unregister_logger, log as _log
from delira.logging.base_logger import make_logger

log = _log


class LoggingContext(object):
    """
    Contextmanager to set a new logging context
    """

    def __init__(
            self,
            name,
            initialize_if_missing=False,
            destroy_on_exit=None,
            **kwargs):
        """

        Parameters
        ----------
        name : str
            the name of the logger to use
        initialize_if_missing : bool
            whether to create a logger if it does not yet exist
        destroy_on_exit : bool
            whether to destroy the logger on exit; If None, the logger will
            only be destroyed, if it was created here
        **kwargs:
            additional keyword arguments to create a logger if necessary

        Raises
        ------
        ValueError
            if the logger does not exist already and shall not be created
        """

        # Logger does exist already
        if logger_exists(name):
            self._name = name
            if destroy_on_exit is None:
                destroy_on_exit = False

        # logger will be created
        elif initialize_if_missing:
            register_logger(make_logger(**kwargs), name)
            if destroy_on_exit is None:
                destroy_on_exit = True
            self._name = name

        # logger does not exist and shall not be created
        else:
            raise ValueError("No valid logger for name %s and "
                             "'initialize_if_missing' is False" % name)

        self._destroy_on_exit = destroy_on_exit

    def __enter__(self):
        """
        Function to be executed during entrance;
        Resets the logging context

        Returns
        -------
        :class:`LoggingContext`
            self
        """
        global log
        log = self.log
        return self

    def __exit__(self, *args):
        """
        Function to be called during exiting the context manager;
        Destroys the logger if necessary and resets the old logging context

        Parameters
        ----------
        *args
            Postional arguments to be compatible with other context managers

        Returns
        -------

        """
        if self._destroy_on_exit:
            _logger = unregister_logger(self._name)
            del _logger

        global log
        log = _log

    def log(self, msg: dict):
        """
        Main Logging Function, Decides whether to log with the assigned
        backend or python's internal module

        Parameters
        ----------
        msg : dict
            the message to log; Should be a dict, where the keys indicate the
            logging function to execute, and the corresponding value holds
            the arguments necessary to execute this function
        """

        _log(msg, self._name)

    def __call__(self, log_message: dict):
        """
        Makes the class callable and forwards the call to
        :meth:`delira.logging.base_logger.Logger.log`

        Parameters
        ----------
        log_message : dict
            the logging message to log

        Returns
        -------
        Any
            the return value obtained by
            :meth:`LoggingContext.log`

        """
        return self.log(log_message)
