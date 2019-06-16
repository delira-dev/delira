from multiprocessing import Queue, Event
from delira.logging.base_backend import BaseBackend
import logging


class Logger(object):
    def __init__(self, backend: BaseBackend, max_queue_size: int = None,
                 level=logging.NOTSET):

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

        # if not self._backend.is_alive():
        #     self._backend.start()

        if self._abort_event.is_set():
            self.close()
            raise RuntimeError("Abort-Event in logging process was set: %s"
                               % self._backend.name)

        if isinstance(log_message, dict):
            self._flush_queue.put_nowait(log_message)
        elif isinstance(log_message, (tuple, list)) and len(log_message) == 2:
            self._flush_queue.put_nowait(log_message)
        else:
            logging.log(self._level, log_message)

    def close(self):
        self._flush_queue.close()
        self._flush_queue.join_thread()

        self._abort_event.set()

    def __del__(self):
        self.close()


class SingleThreadedLogger(Logger):

    def log(self, log_message: dict):
        super().log(log_message)
        self._backend.run()


def make_logger(backend: BaseBackend, max_queue_size: int = None,
                level=logging.NOTSET):

    return SingleThreadedLogger(backend, max_queue_size, level)