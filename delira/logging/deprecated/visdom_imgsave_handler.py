import logging
from delira.utils.decorators import make_deprecated
from .imgsave_handler import ImgSaveHandler
from .visdom_image_handler import VisdomImageHandler


from ..trixi_handler import TrixiHandler


@make_deprecated(TrixiHandler)
class VisdomImageSaveHandler(logging.Handler):
    """
    Logs images to dir and to visdom

    .. deprecated:: 0.1
        :class:`VisdomImageSaveHandler` will be removed in next release and is
        deprecated in favor of ``trixi.logging`` Modules

    ..warning::
        :class:`VisdomImageSaveHandler` will be removed in next release

    See Also
    --------
    `Visdom`
    :class:`VisdomImageHandler`
    :class:`TrixiHandler`

    """
    def __init__(self, port, prefix, log_freq_train, log_freq_val=int(1e10),
                 save_dir_train="./images_train", save_dir_val="./images_val",
                 save_freq_train=1, save_freq_val=1, log_freq_img=1,
                 level=logging.NOTSET, **kwargs):
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
        save_dir_val: str (default:None)
            path to which the training images should be saved (must not yet be
            existent)
        save_freq_train : int (default: 1)
            frequency with which images are saved during training
        save_freq_val : int (default: 1)
            frequency with which images are saved during validation
        level : int (default: logging.NOTSET)
            logging level
        **kwargs:
            additional keyword arguments which are directly passed to visdom

        """
        super().__init__(level)

        self._visdom_handler = VisdomImageHandler(port, prefix, log_freq_train,
                                                  log_freq_val, level,
                                                  log_freq_img=log_freq_img,
                                                  **kwargs)
        self._img_saver = ImgSaveHandler(save_dir_train=save_dir_train,
                                         save_dir_val=save_dir_val,
                                         save_freq_train=save_freq_train,
                                         save_freq_val=save_freq_val,
                                         level=level)

    def emit(self, record):
        """
        log images to visdom and dir

        Parameters
        ----------
        record : LogRecord
            entities to log

        """
        self._visdom_handler.emit(record)
        self._img_saver.emit(record)
