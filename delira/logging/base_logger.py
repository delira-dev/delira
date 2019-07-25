from multiprocessing import Queue, Event
from queue import Full
from delira.logging.base_backend import BaseBackend
import logging


class Logger(object):
    """
    The actual Logger Frontend, passing logging messages to the assigned
    logging backend if appropriate or to python's logging module if not
    """

    def __init__(self, backend: BaseBackend, max_queue_size: int = None,
                 level=logging.NOTSET):
        """

        Parameters
        ----------
        backend : :class:`delira.logging.base_backend.BaseBackend`
            the logging backend to use
        max_queue_size : int
            the maximum size for the queue; if queue is full, all additional
            logging tasks will be dropped until some tasks inside the queue
            were executed; Per default no maximum size is applied
        level : int
            the logging value to use if passing the logging message to
            python's logging module because it is not appropriate for logging
            with the assigned logging backend
        """

        # 0 means unlimited size, but None is more readable
        if max_queue_size is None:
            max_queue_size = 0
        self._abort_event = Event()
        self._flush_queue = Queue(max_queue_size)
        self._backend = backend
        self._backend.set_queue(self._flush_queue)
        self._backend.set_event(self._abort_event)
        self._level = level

    def log(self, log_message: dict):
        """
        Main Logging Function, Decides whether to log with the assigned
        backend or python's internal module

        Parameters
        ----------
        log_message : dict
            the message to log; Should be a dict, where the keys indicate the
            logging function to execute, and the corresponding value holds
            the arguments necessary to execute this function

        Raises
        ------
        RuntimeError
            If the abort event was set externally

        """

        try:
            if self._abort_event.is_set():
                self.close()
                raise RuntimeError("Abort-Event in logging process was set: %s"
                                   % self._backend.name)

            # convert tuple to dict if necessary
            if isinstance(log_message, (tuple, list)):
                if len(log_message) == 2:
                    log_message = (log_message, )
                log_message = dict(log_message)

            # try logging and drop item if queue is full
            try:
                # logging appropriate message with backend
                if isinstance(log_message, dict):
                    self._flush_queue.put_nowait(log_message)
                else:
                    # logging inappropriate message with python's logging
                    logging.log(self._level, log_message)
            except Full:
                pass

        # if an exception was raised anywhere, the abort event will be set
        except Exception as e:
            self._abort_event.set()
            raise e

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
            :meth:`delira.logging.base_logger.Logger.log`

        """
        return self.log(log_message)

    def close(self):
        """
        Function to close the actual logger; Waits for queue closing and sets
        the abortion event

        """
        self._flush_queue.close()
        self._flush_queue.join_thread()

        self._abort_event.set()

    def __del__(self):
        """
        Function to be executed, when class instance will be deleted;
        Calls :meth:`delira.logging.base_logger.Logger.close`

        """

        self.close()


class SingleThreadedLogger(Logger):
    """
    A single threaded Logger which executes the backend after logging
    a single element
    """

    def log(self, log_message: dict):
        """
        Function to log an actual logging message; Calls the backend to
        execute the logging right after pushing it to the queue

        Parameters
        ----------
        log_message : dict
            the message to log; Should be a dict, where the keys indicate the
            logging function to execute, and the corresponding value holds
            the arguments necessary to execute this function

        """
        super().log(log_message)
        self._backend.run()


def make_logger(backend: BaseBackend, max_queue_size: int = None,
                level=logging.NOTSET):
    """
    Function to create a logger

    Parameters
    ----------
    backend : :class:`delira.logging.base_backend.BaseBackend`
        the logging backend
    max_queue_size : int
        the maximum queue size
    level : int
        the logging level for python's internal logging module

    Notes
    -----
    This function shall be used to create
    Loggers (if possible), since it may be extended with new functionalities
    in the future

    Returns
    -------
    :class:`Logger`
        the instance of aa newly created logger

    """

    return SingleThreadedLogger(backend, max_queue_size, level)
