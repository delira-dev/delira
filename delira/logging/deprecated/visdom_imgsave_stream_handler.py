import logging
from delira.utils.decorators import make_deprecated
from ..multistream_handler import MultiStreamHandler
from .visdom_imgsave_handler import VisdomImageSaveHandler


from ..trixi_handler import TrixiHandler


@make_deprecated(TrixiHandler)
class VisdomImageSaveStreamHandler(logging.Handler):
    """
    Logs metrics to streams, metric plots and images to visdom

    .. deprecated:: 0.1
        :class:`VisdomImageSaveStreamHandler` will be removed in next release
        and is
        deprecated in favor of ``trixi.logging`` Modules

    .. warning::
        :class:`VisdomImageSaveStreamHandler` will be removed in next release

    See Also
    --------
    `Visdom`
    :class:`VisdomImageHandler`
    :class:`MultiStreamHandler`
    :class:`TrixiHandler`

    """
    def __init__(self, port, prefix, log_freq_train, log_freq_val=1e10,
                 save_dir_train="./images_train", save_dir_val="./images_val",
                 save_freq_train=1, save_freq_val=1, log_freq_img=1,
                 streams=[], level=logging.NOTSET, **kwargs):
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
        save_dir_train : str
            path to which the training images should be saved (must not yet be
            existent)
        save_dir_val : string (default:None)
            path to which the training images should be saved (must not yet be
            existent)
        save_freq_train : int (default: 1)
            frequency with which images are saved during training
        save_freq_val : int (default: 1)
            frequency with which images are saved during validation
        streams : list of streams with write()-attribute
            streams which are passed to StreamHandlers
        level : int (default: logging.NOTSET)
            logging level
        **kwargs :
            additional keyword arguments which are directly passed to visdom

        """
        super().__init__(level)

        self._visdom_image_handler = VisdomImageSaveHandler(
            port, prefix, log_freq_train, log_freq_val=log_freq_val,
            save_freq_train=save_freq_train, save_freq_val=save_freq_val,
            save_dir_train=save_dir_train, save_dir_val=save_dir_val,
            log_freq_img=log_freq_img, level=level, **kwargs)

        self._stream_handler = MultiStreamHandler(*streams, level=level)

    def emit(self, record: dict):
        """
        Logs metrics to streams, metric plots and images to visdom

        Parameters
        ----------
        record: dict
            entities to log

        """
        if isinstance(record.msg, dict):
            self._visdom_image_handler.emit(record)
            _, _ = record.msg.pop("scalars", {}), record.msg.pop("images", {})

        if record.msg:
            self._stream_handler.emit(record)
