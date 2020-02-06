import typing
import logging
import pickle
import os
from datetime import datetime
import warnings

import copy

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, \
    StratifiedShuffleSplit, ShuffleSplit

from delira import get_backends

from delira.data_loading import DataManager
from delira.models import AbstractNetwork

from delira.utils import DeliraConfig
from delira.training.base_trainer import BaseNetworkTrainer
from delira.training.predictor import Predictor

logger = logging.getLogger(__name__)


class BaseExperiment(object):
    """
    Baseclass for Experiments.

    Implements:

    * Setup-Behavior for Models, Trainers and Predictors (depending on train
        and test case)

    * The K-Fold logic (including stratified and random splitting)

    * Argument Handling

    """

    def __init__(self,
                 config: typing.Union[str, DeliraConfig],
                 model_cls: AbstractNetwork,
                 n_epochs=None,
                 name=None,
                 save_path=None,
                 key_mapping=None,
                 val_score_key=None,
                 optim_builder=None,
                 checkpoint_freq=1,
                 trainer_cls=BaseNetworkTrainer,
                 predictor_cls=Predictor,
                 **kwargs):
        """

        Parameters
        ----------
        config : :class:`DeliraConfig` or str
            the training parameters, if string is passed,
            it is treated as a path to a file, where the
            config is loaded from
        model_cls : Subclass of :class:`AbstractNetwork`
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
            prediction with :class:`Predictor`-API)
        val_score_key : str or None
            key defining which metric to use for validation (determining best
            model and scheduling lr); if None: No validation-based operations
            will be done (model might still get validated, but validation
            metrics
            can only be logged and not used further)
        optim_builder : function
            Function returning a dict of backend-specific optimizers
        checkpoint_freq : int
            frequency of saving checkpoints (1 denotes saving every epoch,
            2 denotes saving every second epoch etc.); default: 1
        trainer_cls : subclass of :class:`BaseNetworkTrainer`
            the trainer class to use for training the model
        predictor_cls : subclass of :class:`Predictor`
            the predictor class to use for testing the model
        **kwargs :
            additional keyword arguments

        """

        # config could also be a file containing config information
        if isinstance(config, str):
            config = config.create_from_file(config)

        if n_epochs is None:
            n_epochs = config.nested_get("n_epochs",
                                         config.nested_get("num_epochs"))

        self.n_epochs = n_epochs

        if name is None:
            name = "UnnamedExperiment"
        self.name = name

        if save_path is None:
            save_path = os.path.abspath(".")

        self.save_path = os.path.join(save_path, name,
                                      str(datetime.now().strftime(
                                          "%y-%m-%d_%H-%M-%S")))

        if os.path.isdir(self.save_path):
            logger.warning("Save Path %s already exists")

        os.makedirs(self.save_path, exist_ok=True)

        self.trainer_cls = trainer_cls
        self.predictor_cls = predictor_cls

        if val_score_key is None:
            warnings.warn("No 'val_score_key' is given. This disables the "
                          "automatic selection of the best model",
                          UserWarning)

        self.val_score_key = val_score_key

        assert key_mapping is not None
        self.key_mapping = key_mapping

        self.config = config
        self.model_cls = model_cls

        self._optim_builder = optim_builder
        self.checkpoint_freq = checkpoint_freq

        self._run = 0

        self.kwargs = kwargs

    def setup(self, config, training=True, **kwargs):
        """
        Defines the setup behavior (model, trainer etc.) for training and
        testing case

        Parameters
        ----------
        config : :class:`DeliraConfig`
            the config to use for setup
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
        if training:
            return self._setup_training(config, **kwargs)

        return self._setup_test(config, **kwargs)

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

        model = self.model_cls(**model_kwargs)

        training_params = config.training_params

        losses = training_params.nested_get("losses")
        optimizer_cls = training_params.nested_get("optimizer_cls")
        optimizer_params = training_params.nested_get("optimizer_params")
        train_metrics = training_params.nested_get("train_metrics", {})
        lr_scheduler_cls = training_params.nested_get("lr_sched_cls", None)
        lr_scheduler_params = training_params.nested_get("lr_sched_params",
                                                         {})

        metrics = training_params.nested_get("metrics", {})

        # ToDo: remove after next release
        val_metrics = config.nested_get("val_metrics", {})
        train_metrics = config.nested_get("train_metrics", {})

        if val_metrics or train_metrics:
            warnings.warn("'val_metrics' and 'train_metrics' are deprecated. "
                          "Please use the combined 'metrics' instead!",
                          DeprecationWarning)
            metrics.update(val_metrics)
            metrics.update(train_metrics)

        # necessary for resuming training from a given path
        save_path = kwargs.pop("save_path", os.path.join(
            self.save_path,
            "checkpoints",
            "run_%02d" % self._run))

        return self.trainer_cls(
            network=model,
            save_path=save_path,
            losses=losses,
            key_mapping=self.key_mapping,
            optimizer_cls=optimizer_cls,
            optimizer_params=optimizer_params,
            train_metrics=train_metrics,
            metrics=metrics,
            lr_scheduler_cls=lr_scheduler_cls,
            lr_scheduler_params=lr_scheduler_params,
            optim_fn=self._optim_builder,
            save_freq=self.checkpoint_freq,
            **kwargs
        )

    def _setup_test(self, config, model, convert_batch_to_npy_fn,
                    prepare_batch_fn, **kwargs):
        """

        Parameters
        ----------
        config : :class:`DeliraConfig`
            the parameters containing the model and training kwargs
            (ignored here, just passed for subclassing and unified API)
        model : :class:`AbstractNetwork`
            the model to test
        convert_batch_to_npy_fn : function
            function to convert a batch of tensors to numpy
        prepare_batch_fn : function
            function to convert a batch-dict to a format accepted by the model.
            This conversion typically includes dtype-conversion, reshaping,
            wrapping to backend-specific tensors and pushing to correct devices
        **kwargs :
            additional keyword arguments

        Returns
        -------
        :class:`Predictor`
            the created predictor

        """
        predictor = self.predictor_cls(
            model=model, key_mapping=self.key_mapping,
            convert_batch_to_npy_fn=convert_batch_to_npy_fn,
            prepare_batch_fn=prepare_batch_fn, **kwargs)
        return predictor

    def run(self, train_data: DataManager,
            val_data: DataManager = None,
            config: DeliraConfig = None, **kwargs):
        """
        Setup and run training

        Parameters
        ----------
        train_data : :class:`DataManager`
            the data to use for training
        val_data : :class:`DataManager` or None
            the data to use for validation (no validation is done
            if passing None); default: None
        config : :class:`DeliraConfig` or None
            the config to use for training and model instantiation
            (will be merged with ``self.config``)
        **kwargs :
            additional keyword arguments

        Returns
        -------
        :class:`AbstractNetwork`
            The trained network returned by the trainer (usually best network)

        See Also
        --------
        :class:`BaseNetworkTrainer` for training itself

        """

        config = self._resolve_params(config)
        kwargs = self._resolve_kwargs(kwargs)

        training_params = config.training_params

        trainer = self.setup(config, training=True, **kwargs)

        self._run += 1

        num_epochs = kwargs.get("num_epochs", training_params.nested_get(
            "num_epochs", self.n_epochs))

        if num_epochs is None:
            num_epochs = self.n_epochs

        return trainer.train(num_epochs, train_data, val_data,
                             self.val_score_key, kwargs.get("val_score_mode",
                                                            "lowest"))

    def resume(self, save_path: str, train_data: DataManager,
               val_data: DataManager = None,
               config: DeliraConfig = None, **kwargs):
        """
        Resumes a previous training by passing an explicit ``save_path``
        instead of generating a new one

        Parameters
        ----------
        save_path : str
            path to previous training
        train_data : :class:`DataManager`
            the data to use for training
        val_data : :class:`DataManager` or None
            the data to use for validation (no validation is done
            if passing None); default: None
        config : :class:`DeliraConfig` or None
            the config to use for training and model instantiation
            (will be merged with ``self.config``)
        **kwargs :
            additional keyword arguments

        Returns
        -------
        :class:`AbstractNetwork`
            The trained network returned by the trainer (usually best network)

        See Also
        --------
        :class:`BaseNetworkTrainer` for training itself

        """
        return self.run(
            train_data=train_data,
            val_data=val_data,
            config=config,
            save_path=save_path,
            **kwargs)

    def test(self, network, test_data: DataManager,
             metrics: dict, metric_keys=None,
             verbose=False, prepare_batch=None,
             convert_fn=lambda *x, **y: (x, y), **kwargs):
        """
        Setup and run testing on a given network

        Parameters
        ----------
        network : :class:`AbstractNetwork`
            the (trained) network to test
        test_data : :class:`DataManager`
            the data to use for testing
        metrics : dict
            the metrics to calculate
        metric_keys : dict of tuples
            the batch_dict keys to use for each metric to calculate.
            Should contain a value for each key in ``metrics``.
            If no values are given for a key, per default ``pred`` and
            ``label`` will be used for metric calculation
        verbose : bool
            verbosity of the test process
        prepare_batch : function
            function to convert a batch-dict to a format accepted by the model.
            This conversion typically includes dtype-conversion, reshaping,
            wrapping to backend-specific tensors and pushing to correct devices
        convert_fn : function
            function to convert a batch of tensors to numpy
        **kwargs :
            additional keyword arguments

        Returns
        -------
        dict
            all predictions obtained by feeding the ``test_data`` through the
            ``network``
        dict
            all metrics calculated upon the ``test_data`` and the obtained
            predictions

        """

        kwargs = self._resolve_kwargs(kwargs)

        predictor = self.setup(None, training=False, model=network,
                               convert_batch_to_npy_fn=convert_fn,
                               prepare_batch_fn=prepare_batch, **kwargs)

        # return first item of generator
        return next(predictor.predict_data_mgr_cache_all(test_data, 1, metrics,
                                                         metric_keys, verbose))

    def kfold(self, data: DataManager, metrics: dict, num_epochs=None,
              num_splits=None, shuffle=False, random_seed=None,
              split_type="random", val_split=0.2, label_key="label",
              train_kwargs: dict = None, metric_keys: dict = None,
              test_kwargs: dict = None, config=None, verbose=False, **kwargs):
        """
        Performs a k-Fold cross-validation

        Parameters
        ----------
        data : :class:`DataManager`
            the data to use for training(, validation) and testing. Will be
            split based on ``split_type`` and ``val_split``
        metrics : dict
            dictionary containing the metrics to evaluate during k-fold
        num_epochs : int or None
            number of epochs to train (if not given, will either be extracted
            from ``config``, ``self.config`` or ``self.n_epochs``)
        num_splits : int or None
            the number of splits to extract from ``data``.
            If None: uses a default of 10
        shuffle : bool
            whether to shuffle the data before splitting or not (implemented by
            index-shuffling rather than actual data-shuffling to retain
            potentially lazy-behavior of datasets)
        random_seed : None
            seed to seed numpy, the splitting functions and the used
            backend-framework
        split_type : str
            must be one of ['random', 'stratified']
            if 'random': uses random data splitting
            if 'stratified': uses stratified data splitting. Stratification
            will be based on ``label_key``
        val_split : float or None
            the fraction of the train data to use as validation set. If None:
            No validation will be done during training; only testing for each
            fold after the training is complete
        label_key : str
            the label to use for stratification. Will be ignored unless
            ``split_type`` is 'stratified'. Default: 'label'
        train_kwargs : dict or None
            kwargs to update the behavior of the :class:`DataManager`
            containing the train data. If None: empty dict will be passed
        metric_keys : dict of tuples
            the batch_dict keys to use for each metric to calculate.
            Should contain a value for each key in ``metrics``.
            If no values are given for a key, per default ``pred`` and
            ``label`` will be used for metric calculation
        test_kwargs : dict or None
            kwargs to update the behavior of the :class:`DataManager`
            containing the test and validation data.
            If None: empty dict will be passed
        config : :class:`DeliraConfig`or None
            the training and model parameters
            (will be merged with ``self.config``)
        verbose : bool
            verbosity
        **kwargs :
            additional keyword arguments

        Returns
        -------
        dict
            all predictions from all folds
        dict
            all metric values from all folds

        Raises
        ------
        ValueError
            if ``split_type`` is neither 'random', nor 'stratified'

        See Also
        --------

        * :class:`sklearn.model_selection.KFold`
        and :class:`sklearn.model_selection.ShuffleSplit`
        for random data-splitting

        * :class:`sklearn.model_selection.StratifiedKFold`
        and :class:`sklearn.model_selection.StratifiedShuffleSplit`
        for stratified data-splitting

        * :meth:`DataManager.update_from_state_dict` for updating the
        data managers by kwargs

        * :meth:`BaseExperiment.run` for the training

        * :meth:`BaseExperiment.test` for the testing

        Notes
        -----
        using stratified splits may be slow during split-calculation, since
        each item must be loaded once to obtain the labels necessary for
        stratification.

        """

        # set number of splits if not specified
        if num_splits is None:
            num_splits = 10
            logger.warning("num_splits not defined, using default value of \
                                    10 splits instead ")

        metrics_test, outputs = {}, {}
        split_idxs = list(range(len(data.dataset)))

        if train_kwargs is None:
            train_kwargs = {}
        if test_kwargs is None:
            test_kwargs = {}

        # switch between differnt kfold types
        if split_type == "random":
            split_cls = KFold
            val_split_cls = ShuffleSplit
            # split_labels are ignored for random splitting, set them to
            # split_idxs just ensures same length
            split_labels = split_idxs
        elif split_type == "stratified":
            split_cls = StratifiedKFold
            val_split_cls = StratifiedShuffleSplit
            # iterate over dataset to get labels for stratified splitting
            split_labels = [data.dataset[_idx][label_key]
                            for _idx in split_idxs]
        else:
            raise ValueError("split_type must be one of "
                             "['random', 'stratified'], but got: %s"
                             % str(split_type))

        fold = split_cls(n_splits=num_splits, shuffle=shuffle,
                         random_state=random_seed)

        if random_seed is not None:
            np.random.seed(random_seed)

        # iterate over folds
        for idx, (train_idxs, test_idxs) in enumerate(
                fold.split(split_idxs, split_labels)):

            # extract data from single manager
            train_data = data.get_subset(train_idxs)
            test_data = data.get_subset(test_idxs)

            train_data.update_state_from_dict(copy.deepcopy(train_kwargs))
            test_data.update_state_from_dict(copy.deepcopy(test_kwargs))

            val_data = None
            if val_split is not None:
                if split_type == "random":
                    # split_labels are ignored for random splitting, set them
                    # to split_idxs just ensures same length
                    train_labels = train_idxs
                elif split_type == "stratified":
                    # iterate over dataset to get labels for stratified
                    # splitting
                    train_labels = [train_data.dataset[_idx][label_key]
                                    for _idx in train_idxs]
                else:
                    raise ValueError("split_type must be one of "
                                     "['random', 'stratified'], but got: %s"
                                     % str(split_type))

                _val_split = val_split_cls(n_splits=1, test_size=val_split,
                                           random_state=random_seed)

                for _train_idxs, _val_idxs in _val_split.split(train_idxs,
                                                               train_labels):
                    val_data = train_data.get_subset(_val_idxs)
                    val_data.update_state_from_dict(copy.deepcopy(test_kwargs))

                    train_data = train_data.get_subset(_train_idxs)

            model = self.run(train_data=train_data, val_data=val_data,
                             config=config, num_epochs=num_epochs, fold=idx,
                             **kwargs)

            _outputs, _metrics_test = self.test(model, test_data,
                                                metrics=metrics,
                                                metric_keys=metric_keys,
                                                verbose=verbose)

            outputs[str(idx)] = _outputs
            metrics_test[str(idx)] = _metrics_test

        return outputs, metrics_test

    def __str__(self):
        """
        Converts :class:`BaseExperiment` to string representation

        Returns
        -------
        str
            representation of class

        """
        s = "Experiment:\n"
        for k, v in vars(self).items():
            s += "\t{} = {}\n".format(k, v)
        return s

    def __call__(self, *args, **kwargs):
        """
        Call :meth:`BaseExperiment.run`

        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments

        Returns
        -------
        :class:`BaseNetworkTrainer`
            trainer of trained network

        """
        return self.run(*args, **kwargs)

    def save(self):
        """
        Saves the Whole experiments

        """
        with open(os.path.join(self.save_path, "experiment.delira.pkl"),
                  "wb") as f:
            pickle.dump(self, f)

        self.config.dump(os.path.join(self.save_path, "parameters"))

    @staticmethod
    def load(file_name):
        """
        Loads whole experiment

        Parameters
        ----------
        file_name : str
            file_name to load the experiment from

        """
        with open(file_name, "rb") as f:
            return pickle.load(f)

    def _resolve_params(self, config: typing.Union[DeliraConfig, None]):
        """
        Merges the given config with ``self.config``.
        If the same argument is given in both configs,
        the one from the currently given config is used here

        Parameters
        ----------
        config : :class:`DeliraConfig` or None
            the parameters to merge with ``self.config``


        Returns
        -------
        :class:`Parameters`
            the merged parameter instance

        """
        if config is None:
            config = DeliraConfig()

        if hasattr(self, "config") and isinstance(self.config, DeliraConfig):
            _config = copy.deepcopy(config)
            config = self.config
            config.update(_config, overwrite=True)

        return config

    def _resolve_kwargs(self, kwargs: typing.Union[dict, None]):
        """
        Merges given kwargs with ``self.kwargs``
        If same argument is present in both kwargs, the one from the given
        kwargs will be used here

        Parameters
        ----------
        kwargs : dict
            the given kwargs to merge with self.kwargs

        Returns
        -------
        dict
            merged kwargs

        """

        if kwargs is None:
            kwargs = {}

        if hasattr(self, "kwargs") and isinstance(self.kwargs, dict):
            _kwargs = kwargs
            kwargs = self.kwargs
            kwargs.update(_kwargs)

        return kwargs

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        vars(self).update(state)
