from multiprocessing.queues import Queue as MpQueue
from threading import Event
from queue import Queue, Full
from delira.logging.base_backend import BaseBackend
from delira.utils.dict_reductions import get_reduction, possible_reductions, \
    reduce_dict
import logging
from types import FunctionType


class Logger(object):
    """
    The actual Logger Frontend, passing logging messages to the assigned
    logging backend if appropriate or to python's logging module if not
    """

    def __init__(self, backend: BaseBackend, max_queue_size: int = None,
                 logging_frequencies=None, reduce_types=None,
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
        logging_frequencies : int or dict
            specifies how often to log for each key.
            If int: integer will be applied to all valid keys
            if dict: should contain a frequency per valid key. Missing keys
            will be filled with a frequency of 1 (log every time)
            None is equal to empty dict here.
        reduce_types : str of FunctionType or dict
            Values are logged in each iteration. This argument specifies,
            how to reduce them to a single value if a logging_frequency
            besides 1 is passed

            if str:
                specifies the reduction type to use. Valid types are
                'last' | 'first' | 'mean' | 'median' | 'max' | 'min'.
                The given type will be mapped to all valid keys.
            if FunctionType:
                specifies the actual reduction function. Will be applied for
                all keys.
            if dict: should contain pairs of valid logging keys and either str
                or FunctionType. Specifies the logging value per key.
                Missing keys will be filles with a default value of 'last'.
                Valid types for strings are
                'last' | 'first' | 'mean' | 'max' | 'min'.
        level : int
            the logging value to use if passing the logging message to
            python's logging module because it is not appropriate for logging
            with the assigned logging backendDict[str, Callable]

        Warnings
        --------
        Since the intermediate values between to logging steps  are stored in
        memory to enable reduction, this might cause OOM errors easily
        (especially if the logged items are still on GPU).
        If this occurs you may want to choose a lower logging frequency.

        """

        # 0 means unlimited size, but None is more readable
        if max_queue_size is None:
            max_queue_size = 0

        # convert to empty dict if None
        if logging_frequencies is None:
            logging_frequencies = {}

        # if int: assign int to all possible keys
        if isinstance(logging_frequencies, int):
            logging_frequencies = {
                k: logging_frequencies
                for k in backend.KEYWORD_FN_MAPPING.keys()}
        # if dict: update missing keys with 1 and make sure other values
        # are ints
        elif isinstance(logging_frequencies, dict):
            for k in backend.KEYWORD_FN_MAPPING.keys():
                if k not in logging_frequencies:
                    logging_frequencies[k] = 1
                else:
                    logging_frequencies[k] = int(logging_frequencies[k])
        else:
            raise TypeError("Invalid Type for logging frequencies: %s"
                            % type(logging_frequencies).__name__)

        # assign frequencies and create empty queues
        self._logging_frequencies = logging_frequencies
        self._logging_queues = {}

        default_reduce_type = "last"
        if reduce_types is None:
            reduce_types = default_reduce_type

        # map string and function to all valid keys
        if isinstance(reduce_types, (str, FunctionType)):
            reduce_types = {
                k: reduce_types
                for k in backend.KEYWORD_FN_MAPPING.keys()}

        # should be dict by now!
        if isinstance(reduce_types, dict):
            # check all valid keys for occurences
            for k in backend.KEYWORD_FN_MAPPING.keys():
                # use default reduce type if necessary
                if k not in reduce_types:
                    reduce_types[k] = default_reduce_type
                # check it is either valid string or already function type
                else:
                    if not isinstance(reduce_types, FunctionType):
                        assert reduce_types[k] in possible_reductions()
                        reduce_types[k] = str(reduce_types[k])
                # map all strings to actual functions
                if isinstance(reduce_types[k], str):
                    reduce_types[k] = get_reduction(reduce_types[k])

        else:
            raise TypeError("Invalid Type for logging reductions: %s"
                            % type(reduce_types).__name__)

        self._reduce_types = reduce_types

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
                    log_message = (log_message,)
                log_message = dict(log_message)

            # try logging and drop item if queue is full
            try:
                # logging appropriate message with backend
                if isinstance(log_message, dict):
                    # multiple logging instances at once possible with
                    # different keys
                    for k, v in log_message.items():
                        # append tag if tag is given, because otherwise we
                        # would enqueue same types but different tags in same
                        # queue
                        if "tag" in v:
                            queue_key = k + "." + v["tag"]
                        else:
                            queue_key = k

                        # create queue if necessary
                        if queue_key not in self._logging_queues:
                            self._logging_queues[queue_key] = []

                        # append current message to queue
                        self._logging_queues[queue_key].append({k: v})
                        # check if logging should be executed
                        if (len(self._logging_queues[queue_key])
                                % self._logging_frequencies[k] == 0):
                            # reduce elements inside queue
                            reduce_message = reduce_dict(
                                self._logging_queues[queue_key],
                                self._reduce_types[k])
                            # flush reduced elements
                            self._flush_queue.put_nowait(reduce_message)
                            # empty queue
                            self._logging_queues[queue_key] = []
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
        if hasattr(self, "_flush_queue"):
            if isinstance(self._flush_queue, MpQueue):
                self._flush_queue.close()
                self._flush_queue.join_thread()

        if hasattr(self, "abort_event"):
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
                logging_frequencies=None, reduce_types=None,
                level=logging.NOTSET):
    """
    Function to create a logger

    Parameters
    ----------
    backend : :class:`delira.logging.base_backend.BaseBackend`
        the logging backend
    max_queue_size : int
        the maximum queue size
    logging_frequencies : int or dict
            specifies how often to log for each key.
            If int: integer will be applied to all valid keys
            if dict: should contain a frequency per valid key. Missing keys
            will be filled with a frequency of 1 (log every time)
            None is equal to empty dict here.
    reduce_types : str of FunctionType or dict
        if str:
            specifies the reduction type to use. Valid types are
            'last' | 'first' | 'mean' | 'max' | 'min'.
            The given type will be mapped to all valid keys.
        if FunctionType:
            specifies the actual reduction function. Will be applied for
            all keys.
        if dict: should contain pairs of valid logging keys and either str
            or FunctionType. Specifies the logging value per key.
            Missing keys will be filles with a default value of 'last'.
            Valid types for strings are
            'last' | 'first' | 'mean' | 'max' | 'min'.
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

    return SingleThreadedLogger(backend=backend, max_queue_size=max_queue_size,
                                logging_frequencies=logging_frequencies,
                                reduce_types=reduce_types, level=level)
