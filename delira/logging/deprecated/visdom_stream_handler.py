import logging
from delira.utils.decorators import make_deprecated
from .visdom_image_handler import VisdomImageHandler
from ..multistream_handler import MultiStreamHandler


from ..trixi_handler import TrixiHandler


@make_deprecated(TrixiHandler)
class VisdomStreamHandler(logging.Handler):
    """
    Logs images and metric plots to visdom and scalar values to streams

    .. deprecated:: 0.1
        :class:`VisdomStreamHandler` will be removed in next release and is
        deprecated in favor of ``trixi.logging`` Modules

    .. warning::
        :class:`VisdomStreamHandler` will be removed in next release

    See Also
    --------
    `Visdom`
    :class:`VisdomImageHandler`
    :class:`MultiStreamHandler`
    :class:`TrixiHandler`
    """

    def __init__(self, port, prefix, log_freq_train,
                 log_freq_val=int(1e10), streams=[],
                 level=logging.NOTSET, log_freq_img=1, **kwargs):
        """

        Parameters
        ----------
        port : int
            port of visdom-server
        prefix : str
            prefix of environment names
        log_freq_train : int
            Defines logging frequency for scores in train mode
        log_freq_val : int
            Defines logging frequency for scores in validation mode
        streams : list of streams with write()-attribute
            streams which are passed to StreamHandlers
        level : int (default: logging.NOTSET)
            logging level
        **kwargs :
            additional keyword arguments which are directly passed to visdom

        """
        super().__init__()

        self._visdom_handler = VisdomImageHandler(port, prefix, log_freq_train,
                                                  log_freq_val, level,
                                                  log_freq_img=log_freq_img,
                                                  **kwargs)
        self._stream_handler = MultiStreamHandler(*streams, level=level)
        self.keys_pop = ('scores', 'images', 'heatmaps', 'fold')

    def emit(self, record: dict):
        """
        Logs images and metric plots to visdom and scalar values to streams

        Parameters
        ----------
        record : LogRecord
            entities to log

        """
        if isinstance(record.msg, dict):
            self._visdom_handler.emit(record)

            for key in self.keys_pop:
                _ = record.msg.pop(key, {})

        if record.msg:
            self._stream_handler.emit(record)
