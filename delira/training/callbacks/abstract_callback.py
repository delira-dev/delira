class AbstractCallback(object):
    """
    Implements abstract callback interface.
    All callbacks should be derived from this class

    See Also
    --------
    :class:`AbstractNetworkTrainer`

    """

    def __init__(self, *args, **kwargs):
        """

        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments

        """
        super().__init__(*args, **kwargs)

    def at_epoch_begin(self, trainer, *args, **kwargs):
        """
        Function which will be executed at begin of each epoch

        Parameters
        ----------
        trainer : :class:`AbstractNetworkTrainer`
        **kwargs :
            additional keyword arguments

        Returns
        -------
        dict
            modified trainer attributes, where the name must correspond to the
            trainer's attribute name

        Notes
        -----
        The basetrainer calls the callbacks with the following additional
        arguments: `val_metrics`(dict), `val_score_key`(str), `curr_epoch`(int)
        """
        return {}

    def at_epoch_end(self, trainer, *args, **kwargs):
        """
        Function which will be executed at end of each epoch

        Parameters
        ----------
        trainer : :class:`AbstractNetworkTrainer`
        **kwargs :
            additional keyword arguments

        Returns
        -------
        dict
            modified trainer attributes, where the name must correspond to the
            trainer's attribute name

        Notes
        -----
        The basetrainer calls the callbacks with the following additional
        arguments: `val_metrics`(dict), `val_score_key`(str), `curr_epoch`(int)
        """
        return {}

    def at_training_begin(self, trainer, *args, **kwargs):
        """
        Function which will be executed at begin of training

        Parameters
        ----------
        trainer : :class:`AbstractNetworkTrainer`
        **kwargs :
            additional keyword arguments

        Returns
        -------
        dict
            modified trainer attributes, where the name must correspond to the
            trainer's attribute name
        """
        return {}

    def at_training_end(self, trainer, *args, **kwargs):
        """
        Function which will be executed at end of training

        Parameters
        ----------
        trainer : :class:`AbstractNetworkTrainer`
        **kwargs :
            additional keyword arguments

        Returns
        -------
        dict
            modified trainer attributes, where the name must correspond to the
            trainer's attribute name

        """
        return {}

    def at_iter_begin(self, trainer, *args, **kwargs):
        """
        Function which will be executed at begin of each iteration

        Parameters
        ----------
        trainer : :class:`AbstractNetworkTrainer`
        **kwargs :
            additional keyword arguments

        Returns
        -------
        dict
            modified trainer attributes, where the name must correspond to the
            trainer's attribute name

        Notes
        -----
        The predictor calls the callbacks with the following additional
        arguments: `iter_num`(int), `train`(bool)

        The basetrainer adds following arguments (wrt the predictor):
        `curr_epoch`(int), `global_iter_num`(int)

        """
        return {}

    def at_iter_end(self, trainer, *args, **kwargs):
        """
        Function which will be executed at end of each iteration

        Parameters
        ----------
        trainer : :class:`AbstractNetworkTrainer`
        **kwargs :
            additional keyword arguments

        Returns
        -------
        dict
            modified trainer attributes, where the name must correspond to the
            trainer's attribute name

        Notes
        -----
        The predictor calls the callbacks with the following additional
        arguments: `iter_num`(int), `metrics`(dict),
        `data_dict`(dict, contains prediction and input data),
        `train`(bool)

        The basetrainer adds following arguments (wrt the predictor):
        `curr_epoch`(int), `global_iter_num`(int)

        """
        return {}
