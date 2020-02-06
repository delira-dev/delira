from delira.training.callbacks.abstract_callback import AbstractCallback
from delira.logging import make_logger, BaseBackend
import logging


class DefaultLoggingCallback(AbstractCallback):
    """
    A default Logging backend which logs only the metrics; Should be
    subclassed for additional logging
    """

    def __init__(self, backend: BaseBackend, max_queue_size: int = None,
                 logging_frequencies=None, reduce_types=None,
                 level=logging.NOTSET):
        """

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
                specifies the actual reduction function. Will be applied
                for all keys.
            if dict: should contain pairs of valid logging keys and either
                str or FunctionType. Specifies the logging value per key.
                Missing keys will be filles with a default value of 'last'.
                Valid types for strings are
                'last' | 'first' | 'mean' | 'max' | 'min'.
        level : int
            the logging level for python's internal logging module

        """
        super().__init__()

        self._logger = make_logger(backend=backend,
                                   max_queue_size=max_queue_size,
                                   logging_frequencies=logging_frequencies,
                                   reduce_types=reduce_types, level=level)

    def at_iter_end(self, trainer, iter_num=None, data_dict=None, train=False,
                    **kwargs):
        """
        Function logging the metrics at the end of each iteration

        Parameters
        ----------
        trainer : :class:`BaseNetworkTrainer`
            the current trainer object (unused in this callback)
        iter_num : int
            number of the current iteration inside the current epoch
            (unused in this callback)
        data_dict : dict
            the current data dict (including predictions)
        train: bool
            signals if callback is called by trainer or predictor
        **kwargs :
            additional keyword arguments

        Returns
        -------
        dict
            empty dict, because no state should be updated
        """

        metrics = kwargs.get("metrics", {})

        for k, v in metrics.items():
            self._logger.log({"scalar": {"tag": self.create_tag(k, train),
                                         "scalar_value": v}})

        return {}

    @staticmethod
    def create_tag(tag: str, train: bool):
        if train:
            tag = tag + "_val"
        return tag
