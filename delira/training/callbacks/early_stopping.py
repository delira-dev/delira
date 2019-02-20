from .abstract_callback import AbstractCallback


class EarlyStopping(AbstractCallback):
    """
    Implements Early Stopping as callback

    See Also
    --------
    :class:`AbstractCallback`

    """

    def __init__(self, monitor_key,
                 min_delta=0,
                 patience=0,
                 mode='min'):
        """

        Parameters
        ----------
        monitor_key : str
            the validation key to monitor
        min_delta : float or int
            the minimum difference between the best metric value so far and
            the current one
        patience : int
            number of epochs to wait before stopping training
        mode : str (default: 'min')
            defines the optimum for the monitored value

        """

        super().__init__()

        self.monitor_key = monitor_key,
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode

        if 'min' == mode:
            self.best_metric = float('inf')
        elif 'max' == mode:
            self.best_metric = - float('inf')

        else:
            raise ValueError("Unknown compare mode: Got %s, but expected one "
                             "of ['min', 'max']" % mode)
        self.epochs_waited = 0

    def _is_better(self, metric):
        """
        Helper function to decide whether the current metric is better than
        the best metric so far

        Parameters
        ----------
        metric :
            current metric value

        Returns
        -------
        bool
            whether this metric is the new best metric or not

        """
        if 'min' == self.mode:
            return metric < (self.best_metric - self.min_delta)
        else:
            return metric > (self.best_metric + self.min_delta)

    def at_epoch_end(self, trainer, **kwargs):
        """
        Actual early stopping: Checks at end of each epoch if monitored metric
        is new best and if it hasn't improved over `self.patience` epochs, the
        training will be stopped

        Parameters
        ----------
        trainer : :class:`AbstractNetworkTrainer`
            the trainer whose arguments can be modified
        **kwargs :
            additional keyword arguments

        Returns
        -------
        :class:`AbstractNetworkTrainer`
            trainer with modified attributes

        """
        metric = kwargs.get("val_metrics", {})[self.monitor_key]

        self.epochs_waited += 1 - int(self._is_better(metric))

        if self.epochs_waited >= self.patience:
            stop_training = True
        else:
            stop_training = False

        return {"stop_training": stop_training}
