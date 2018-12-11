from logging import Handler, NOTSET
from trixi.logger import AbstractLogger as AbstractTrixiLogger


TRIXI_PREFIXES = ["show_", "plot_", "save_", "get_", ""]


class TrixiHandler(Handler):
    """
    Handler to integrate the :mod:`trixi` loggers into the :mod:`logging`
    module

    """

    def __init__(self, logging_cls, level=NOTSET, *args, **kwargs):
        """

        Parameters
        ----------
        logging_cls :
            logging class (must be subclass of `trixi.logger.AbstractLogger`)
        level : int (default: NOTSET)
            logging level
        *args :
            positional arguments to instantiate the logger from the
            `logging_cls`
        **kwargs :
            keyword arguments to instantiate the logger from the `logging_cls`

        """
        super().__init__(level)

        assertion_str = "%s is not a subclass of %s" % (
            logging_cls.__name__, AbstractTrixiLogger.__name__)

        assert issubclass(logging_cls, AbstractTrixiLogger), assertion_str
        self._logger = logging_cls(*args, **kwargs)

    def emit(self, record):
        """
        logs the record entity to `trixi` loggers

        Parameters
        ----------
        record : LogRecord
            record to log

        """
        if not isinstance(record.msg, dict):
            return

        for key, val in record.msg.items():

            for _prefix in TRIXI_PREFIXES:
                if hasattr(self._logger, _prefix + key) and callable(
                        getattr(self._logger, _prefix + key)):

                    if isinstance(val, dict):
                        # get args from val dict
                        args = val.pop("args", [])
                        # combine kwargs from val["kwargs"} and other
                        # key, val pairs in val
                        kwargs = {**val.pop("kwargs", {}),
                                  **val}

                    else:
                        # check if val is iterable
                        try:
                            iter(val)

                            # val is iterable -> use it as args
                            args = val

                        except TypeError:
                            # val is not iterable -> store it in list and use
                            # this list as args
                            args = [val]

                        # val specifies args -> no kwargs given
                        kwargs = {}

                    getattr(self._logger, _prefix + key)(*args, **kwargs)
