from functools import partial
import typing
import os

from sklearn.base import BaseEstimator

from delira.models.backends.sklearn import SklearnEstimator

from delira.training.base_experiment import BaseExperiment
from delira.utils import DeliraConfig

from delira.training.backends.sklearn.trainer import SklearnEstimatorTrainer


class SklearnExperiment(BaseExperiment):
    def __init__(self,
                 config: typing.Union[str, DeliraConfig],
                 model_cls: BaseEstimator,
                 n_epochs=None,
                 name=None,
                 save_path=None,
                 key_mapping=None,
                 val_score_key=None,
                 checkpoint_freq=1,
                 trainer_cls=SklearnEstimatorTrainer,
                 model_wrapper_cls=SklearnEstimator,
                 **kwargs):
        """

        Parameters
        ----------
        config : :class:`DeliraConfig` or str
            the training config, if string is passed,
            it is treated as a path to a file, where the
            config is loaded from
        model_cls : Subclass of :class:`sklearn.base.BaseEstimator`
            the class implementing the model to train (will be wrapped by
            :class:`SkLearnEstimator`)
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
            given, a default key_mapping of {"X": "X"} will be used here
        checkpoint_freq : int
            frequency of saving checkpoints (1 denotes saving every epoch,
            2 denotes saving every second epoch etc.); default: 1
        trainer_cls : subclass of :class:`SkLearnEstimatorTrainer`
            the trainer class to use for training the model, defaults to
            :class:`PyTorchNetworkTrainer`
        model_wrapper_cls : subclass of :class:`SkLearnEstimator`
            class wrapping the actual sklearn model to provide delira
            compatibility
        **kwargs :
            additional keyword arguments

        """

        if key_mapping is None:
            key_mapping = {"X": "X"}

        super().__init__(config=config,
                         model_cls=model_cls,
                         n_epochs=n_epochs,
                         name=name,
                         save_path=save_path,
                         key_mapping=key_mapping,
                         val_score_key=val_score_key,
                         checkpoint_freq=checkpoint_freq,
                         trainer_cls=trainer_cls,
                         **kwargs)
        self._model_wrapper_cls = model_wrapper_cls

    def _setup_training(self, config, **kwargs):
        """
            Handles the setup for training case

            Parameters
            ----------
            config : :class:`DeliraConfig`
                the config containing the model and training kwargs
            **kwargs :
                additional keyword arguments

            Returns
            -------
            :class:`BaseNetworkTrainer`
                the created trainer
        """
        model_kwargs = config.model_params
        model_kwargs = {**model_kwargs["variable"], **model_kwargs["fixed"]}

        _model = self.model_cls(**model_kwargs)
        model = self._model_wrapper_cls(_model)

        training_params = config.training_params
        metrics = training_params.nested_get("metrics")

        # necessary for resuming training from a given path
        save_path = kwargs.pop("save_path", os.path.join(
            self.save_path,
            "checkpoints",
            "run_%02d" % self._run))

        return self.trainer_cls(
            estimator=model,
            save_path=save_path,
            key_mapping=self.key_mapping,
            metrics=metrics,
            save_freq=self.checkpoint_freq,
            **kwargs
        )

    def _setup_test(self, config, model, convert_batch_to_npy_fn,
                    prepare_batch_fn, **kwargs):
        """

        Parameters
        ----------
        config : :class:`DeliraConfig`
            the config containing the model and training kwargs
            (ignored here, just passed for subclassing and unified API)
        model : :class:`sklearn.base.BaseEstimator`
            the model to test
        convert_batch_to_npy_fn : function
            function to convert a batch of tensors to numpy
        prepare_batch_fn : function
            function to convert a batch-dict to a format accepted by the
            model. This conversion typically includes dtype-conversion,
            reshaping, wrapping to backend-specific tensors and pushing to
            correct devices
        **kwargs :
            additional keyword arguments

        Returns
        -------
        :class:`Predictor`
                        the created predictor

                    """
        if not isinstance(model, SklearnEstimator):
            model = SklearnEstimator(model)

        if prepare_batch_fn is None:
            prepare_batch_fn = partial(model.prepare_batch,
                                       input_device="cpu",
                                       output_device="cpu")

        return super()._setup_test(config, model, convert_batch_to_npy_fn,
                                   prepare_batch_fn, **kwargs)
