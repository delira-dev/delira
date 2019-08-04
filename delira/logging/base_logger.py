from multiprocessing import Queue, Event
from queue import Full
from delira.logging.base_backend import BaseBackend
import logging
import numpy as np
from types import FunctionType
from collections import Iterable, MutableMapping


# Reduction Functions
def _reduce_last(items: list):
    """
    Reduction Function returning the last element

    Parameters
    ----------
    items : list
        the items to reduce

    Returns
    -------
    dict
        reduced items

    """
    return items[-1]


def _reduce_first(items: list):
    """
    Reduction Function returning the first element

    Parameters
    ----------
    items : list
        the items to reduce

    Returns
    -------
    dict
        reduced items

    """
    return items[0]


def _reduce_mean(items: list):
    """
    Reduction Function returning the mean element

    Parameters
    ----------
    items : list
        the items to reduce

    Returns
    -------
    dict
        reduced items

    """
    return np.mean(items)


def _reduce_max(items: list):
    """
    Reduction Function returning the max element

    Parameters
    ----------
    items : list
        the items to reduce

    Returns
    -------
    dict
        reduced items

    """
    return np.max(items)


def _reduce_min(items: list):
    """
    Reduction Function returning the min element

    Parameters
    ----------
    items : list
        the items to reduce

    Returns
    -------
    dict
        reduced items

    """
    return np.min(items)


def _flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return type(d)(items)


def _unflatten_dict(dictionary, sep="."):
    return_dict = {}
    for key, value in dictionary.items():
        parts = key.split(sep)
        d = return_dict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return return_dict


def _reduce_dict(items: list, reduce_fn):
    """
    A function to reduce all entries inside a dict

    Parameters
    ----------
    items : list
        a list of dicts to reduce
    reduce_fn : FunctionType
        a function to apply to all non-equal iterables

    Returns
    -------
    dict
        the reduced dict

    """

    result_dict = {}
    # assuming the type of all items is same for all queued logging dicts and
    # all dicts have the same keys

    flattened_dicts = [_flatten_dict(_tmp, sep=".") for _tmp in items]

    # from list of dicts to dict of lists:
    for d in flattened_dicts:
        for k, v in d.items():
            try:
                result_dict[k].append(v)
            except KeyError:
                result_dict[k] = [v]

    for k, v in result_dict.items():
        # check if all items are equal
        if all([_v == v[0] for _v in v[1:]]):
            # use first item since they are equal
            result_dict[k] = v[0]
        else:
            # apply reduce function
            result_dict[k] = reduce_fn(v)

    # unflatten reduced dict
    return _unflatten_dict(result_dict, sep=".")


# string mapping for reduction functions
_REDUCTION_FUNCTIONS = {
    "last": _reduce_last,
    "first": _reduce_first,
    "mean": _reduce_mean,
    "max": _reduce_max,
    "min": _reduce_min
}


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
            the logging value to use if passing the logging message to
            python's logging module because it is not appropriate for logging
            with the assigned logging backend

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
        self._logging_queues = {
            k: [] for k in backend.KEYWORD_FN_MAPPING.keys()}

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
                        assert reduce_types[k] in _REDUCTION_FUNCTIONS.keys()
                        reduce_types[k] = str(reduce_types[k])
                # map all strings to actual functions
                if isinstance(reduce_types[k], str):
                    reduce_types[k] = _REDUCTION_FUNCTIONS[reduce_types[k]]

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
                        # append current message to queue
                        self._logging_queues[k].append({k: v})
                        # check if logging should be executed
                        if (len(self._logging_queues[k])
                                % self._logging_frequencies[k] == 0):
                            # reduce elements inside queue
                            reduce_message = _reduce_dict(
                                self._logging_queues[k], self._reduce_types[k])
                            # flush reduced elements
                            self._flush_queue.put_nowait(reduce_message)
                            # empty queue
                            self._logging_queues[k] = []
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
