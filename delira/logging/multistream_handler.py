import logging


class MultiStreamHandler(logging.Handler):
    """
    Logging Handler which accepts multiple streams and creates StreamHandlers

    """
    def __init__(self, *streams, level=logging.NOTSET):
        """

        Parameters
        ----------
        streams : streams with write()-attribute
            streams which are passed to StreamHandlers
        level : int (default: logging.NOTSET)
            logging level
        """
        super().__init__(level)

        self._stream_handlers = []

        for _stream in streams:
            self._stream_handlers.append(logging.StreamHandler(_stream))

    def emit(self, record):
        """
        logs the record entity to streams

        Parameters
        ----------
        record : LogRecord
            record to log

        """

        for _handler in self._stream_handlers:
            _handler.emit(record)
