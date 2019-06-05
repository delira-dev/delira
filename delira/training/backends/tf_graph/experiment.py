class TfGraphExperiment(TfEagerExperiment):
    def __init__(self,
                 params: typing.Union[str, Parameters],
                 model_cls: AbstractTfEagerNetwork,
                 n_epochs=None,
                 name=None,
                 save_path=None,
                 key_mapping=None,
                 val_score_key=None,
                 optim_builder=create_optims_default,
                 checkpoint_freq=1,
                 trainer_cls=TfGraphNetworkTrainer,
                 **kwargs):
        """

        Parameters
        ----------
        params : :class:`Parameters` or str
            the training parameters, if string is passed,
            it is treated as a path to a pickle file, where the
            parameters are loaded from
        model_cls : Subclass of :class:`AbstractTfEagerNetwork`
            the class implementing the model to train
        n_epochs : int or None
            the number of epochs to train, if None: can be specified later
            during actual training
        name : str or None
            the Experiment's name
        save_path : str or None
            the path to save the results and checkpoints to.
            if None: Current working directory will be used
        key_mapping : dict
            mapping between data_dict and model inputs (necessary for
            prediction with :class:`Predictor`-API), if no keymapping is
            given, a default key_mapping of {"x": "data"} will be used
            here
        val_score_key : str or None
            key defining which metric to use for validation (determining
            best model and scheduling lr); if None: No validation-based
            operations will be done (model might still get validated,
            but validation metrics can only be logged and not used further)
        optim_builder : function
            Function returning a dict of backend-specific optimizers.
            defaults to :func:`create_optims_default_tf`
        checkpoint_freq : int
            frequency of saving checkpoints (1 denotes saving every epoch,
            2 denotes saving every second epoch etc.); default: 1
        trainer_cls : subclass of :class:`TfEagerNetworkTrainer`
            the trainer class to use for training the model, defaults to
            :class:`TfEagerNetworkTrainer`
        **kwargs :
            additional keyword arguments

        """

        if key_mapping is None:
            key_mapping = {"images": "data"}

            super().__init__(
                params=params,
                model_cls=model_cls,
                n_epochs=n_epochs,
                name=name,
                save_path=save_path,
                key_mapping=key_mapping,
                val_score_key=val_score_key,
                optim_builder=optim_builder,
                checkpoint_freq=checkpoint_freq,
                trainer_cls=trainer_cls,
                **kwargs)

    def setup(self, params, training=True, **kwargs):
        """
        Defines the setup behavior (model, trainer etc.) for training and
        testing case

        Parameters
        ----------
        params : :class:`Parameters`
            the parameters to use for setup
        training : bool
            whether to setup for training case or for testing case
        **kwargs :
            additional keyword arguments

        Returns
        -------
        :class:`BaseNetworkTrainer`
            the created trainer (if ``training=True``)
        :class:`Predictor`
            the created predictor (if ``training=False``)

        See Also
        --------

        * :meth:`BaseExperiment._setup_training` for training setup

        * :meth:`BaseExperiment._setup_test` for test setup

        """
        tf.reset_default_graph()
        return super().setup(self, params=params, training=training,
                             **kwargs)