from delira.training.callbacks.abstract_callback import AbstractCallback


class ReporterCallback(AbstractCallback):
    """
    A ReporterCallback which adds the metrics to the internal state and also
    handles a stopping criterion (which will be passed from the scheduler)
    """

    def __init__(self, stopping_fn):
        """

        Parameters
        ----------
        stopping_fn : function
            A function, which accepts only the reporter as argument and
            returns, whether to stop the current trial; Typically passed by the
            scheduler

        """
        super().__init__()
        self._metrics = {}
        self._curr_epoch = None

        self.stopping_fn = stopping_fn

    def at_epoch_end(self, trainer, val_metrics, val_score_key, curr_epoch,
                     **kwargs):
        """
        function, which will be executed at end of each epoch.

        Parameters
        ----------
        trainer : :class:`BaseNetworkTrainer`
            the actual training state
        val_metrics : dict
            the current validation metrics
        val_score_key : str
            the current validation score key (necessary to determine best
            model)
        curr_epoch : int
            the current epoch
        **kwargs :
            additional keyword arguments

        Returns
        -------
        dict
            the attributes of the trainer which shall be updated

        """

        for key, val in val_metrics.items():
            if key in self._metrics:
                self._metrics[key].append(val)
            else:
                self._metrics[key] = [val]

        self._curr_epoch = curr_epoch

        update_dict = {"stop_training": self.stopping_fn(self)}
        return update_dict
