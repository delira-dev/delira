from delira.logging.tensorboard_backend import TensorboardBackend
from delira.logging.visdom_backend import VisdomBackend
from delira.logging.base_backend import BaseBackend
from delira.logging.writer_backend import WriterLoggingBackend
from delira.logging.base_logger import Logger, SingleThreadedLogger, \
    make_logger
from delira.logging.registry import unregister_logger, register_logger, \
    get_logger, logger_exists, log as _log, get_available_loggers
from delira.logging.logging_context import LoggingContext

log = _log
