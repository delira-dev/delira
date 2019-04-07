

from ..utils import now
from ..data_loading import BaseDataManager, BaseLazyDataset
from delira import __version__ as delira_version
from .parameters import Parameters
from ..models import AbstractNetwork
from .base_trainer import BaseNetworkTrainer
from .predictor import Predictor
from trixi.experiment import Experiment as TrixiExperiment
import os
import logging
import typing
import numpy as np

import pickle
from abc import abstractmethod
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from datetime import datetime
from inspect import signature
from functools import partial

from delira import get_backends

logger = logging.getLogger(__name__)


class AbstractExperiment(TrixiExperiment):
    """
    Abstract Class Representing a single Experiment (must be subclassed for
    each Backend)

    See Also
    --------
    :class:`PyTorchExperiment`

    """
    @abstractmethod
    def __init__(self, n_epochs, *args, **kwargs):
        """

        Parameters
        ----------
        n_epochs : int
            number of epochs to train
        *args :
            positional arguments
        **kwargs :
            keyword arguments
        """
        super().__init__(n_epochs)
        self._run = 0

    @abstractmethod
    def setup(self, *args, **kwargs):
        """
        Abstract Method to setup a :class:`AbstractNetworkTrainer`

        Raises
        ------
        NotImplementedError
            if not overwritten by subclass

        """
        raise NotImplementedError()

    @abstractmethod
    def run(self, train_data: BaseDataManager,
            val_data: typing.Optional[BaseDataManager] = None,
            params: typing.Optional[Parameters] = None,
            **kwargs):
        """
        trains single model

        Parameters
        ----------
        train_data : :class:`BaseDataManager`
            data manager containing the training data
        val_data : :class:`BaseDataManager`
            data manager containing the validation data
        parameters : :class:`Parameters`, optional
            Class containing all parameters (defaults to None).
            If not specified, the parameters fall back to the ones given during
            class initialization

        Raises
        ------
        NotImplementedError
            If not overwritten in subclass

        """

        raise NotImplementedError()

    def test(self,
             params: Parameters,
             network: AbstractNetwork,
             datamgr_test: BaseDataManager,
             metrics: dict,
             prepare_batch=lambda *x: x,
             convert_fn=lambda *x: x,
             verbose=False,
             ** kwargs):
        """
        Executes prediction for all items in datamgr_test with network

        Parameters
        ----------
        params : :class:`Parameters`
            the parameters to construct a model
        network : :class:'AbstractPyTorchNetwork'
            the network to train
        datamgr_test : :class:'BaseDataManager'
            holds the test data
        **kwargs :
            holds additional keyword arguments
            (which are completly passed to the trainers init)

        Returns
        -------
        np.ndarray
            predictions from batches
        list of np.ndarray
            labels from batches
        dict
            dictionary containing the mean validation metrics and
            the mean loss values
        """

        predictor = Predictor(
            network, convert_fn, prepare_batch)

        val_predictions, val_metrics = predictor.predict_data_mgr(
            datamgr_test, 1, metrics=metrics, verbose=verbose)

        return val_predictions, val_metrics

    def kfold(self, num_epochs: int,
              data: BaseDataManager,
              num_splits=None, shuffle=False, random_seed=None, train_kwargs={},
              test_kwargs={}, **kwargs):
        """
        Runs K-Fold Crossvalidation
        The supported scenario is:

            * passing a single datamanager: the data within the single manager
            will be split and multiple datamanagers will be created holding
            the subsets.

        Parameters
        ----------
        num_epochs : int
            number of epochs to train the model
        data : single :class:`BaseDataManager`
            single datamanager (will be split for crossvalidation)
        num_splits : None or int
            number of splits for kfold
            if None: 10 splits will be validated per default
        shuffle : bool
            whether or not to shuffle indices for kfold
        random_seed : None or int
            random seed used to seed the kfold (if shuffle is true),
            pytorch and numpy
        train_kwargs : dict
            keyword arguments to specify training behavior
        test_kwargs : dict
            keyword arguments to specify testing behavior
        **kwargs :
            additional keyword arguments (completely passed to self.run())

        See Also
        --------
        :method:`BaseDataManager.update_state_from_dict` for valid keys in
            ``train_kwargs`` and ``test_kwargs``

        """

        # set number of splits if not specified
        if num_splits is None:
            num_splits = 10
            logger.warning("num_splits not defined, using default value of \
                            10 splits instead ")

        # extract actual data to be split
        split_data = list(range(len(data.dataset)))

        # instantiate the actual kfold
        fold = KFold(n_splits=num_splits, shuffle=shuffle,
                     random_state=random_seed)

        if random_seed is not None:
            np.random.seed(random_seed)

        # run folds
        for idx, (train_idxs, test_idxs) in enumerate(fold.split(split_data)):

            # extract data from single manager
            train_data = data.get_subset(train_idxs)
            test_data = data.get_subset(test_idxs)

            # update manager behavior for train and test case
            train_data.update_state_from_dict(train_kwargs)
            test_data.update_state_from_dict(test_kwargs)

            self.run(train_data, test_data,
                     num_epochs=num_epochs,
                     fold=idx,
                     **kwargs)

    def stratified_kfold(self, num_epochs: int,
                         data: BaseDataManager,
                         num_splits=None, shuffle=False, random_seed=None,
                         label_key="label", train_kwargs={}, test_kwargs={},
                         **kwargs):
        """
        Runs stratified K-Fold Crossvalidation
        The supported supported scenario is:

            * passing a single datamanager: the data within the single manager
              will be split and multiple datamanagers will be created holding
              the subsets.

        Parameters
        ----------
        num_epochs : int
            number of epochs to train the model
        data : :class:`BaseDataManager`
            single datamanager
            (will be split for crossvalidation)
        num_splits : None or int
            number of splits for kfold
            if None: 10 splits will be validated
        shuffle : bool
            whether or not to shuffle indices for kfold
        random_seed : None or int
            random seed used to seed the kfold (if shuffle is true),
            pytorch and numpy
        label_key : str (default: "label")
            the key to extract the label for stratification from each data
            sample
        train_kwargs : dict
            keyword arguments to specify training behavior
        test_kwargs : dict
            keyword arguments to specify testing behavior
        **kwargs :
            additional keyword arguments (completely passed to self.run())

        See Also
        --------
        :method:`BaseDataManager.update_state_from_dict` for valid keys in
            ``train_kwargs`` and ``test_kwargs``

        """

        if num_splits is None:
            num_splits = 10
            logger.warning("num_splits not defined, using default value of \
                                10 splits instead ")

        if isinstance(data.dataset, BaseLazyDataset):
            logger.warning("A lazy dataset is given for stratified kfold. \
                            Iterating over the dataset to extract labels for \
                            stratification may be a massive overhead")

        split_idxs = list(range(len(data.dataset)))
        split_labels = [data.dataset[_idx][label_key] for _idx in split_idxs]

        fold = StratifiedKFold(n_splits=num_splits, shuffle=shuffle,
                               random_state=random_seed)

        for idx, (train_idxs, test_idxs) in enumerate(fold.split(split_idxs,
                                                                 split_labels)):
            # extract data from single manager
            train_data = data.get_subset(train_idxs)
            test_data = data.get_subset(test_idxs)

            # update manager behavior for train and test case
            train_data.update_state_from_dict(train_kwargs)
            test_data.update_state_from_dict(test_kwargs)
            self.run(train_data, test_data,
                     num_epochs=num_epochs,
                     fold=idx,
                     **kwargs)

    def stratified_kfold_predict(self, num_epochs: int,
                                 data: BaseDataManager,
                                 split_val=0.2,
                                 num_splits=None, shuffle=False, random_seed=None,
                                 label_key="label", train_kwargs={}, test_kwargs={},
                                 **kwargs):
        """
        Runs stratified K-Fold Crossvalidation
        The supported supported scenario is:

            * passing a single datamanager: the data within the single manager
              will be split and multiple datamanagers will be created holding
              the subsets.

        Parameters
        ----------
        num_epochs : int
            number of epochs to train the model
        data : :class:`BaseDataManager`
            single datamanager
            (will be split for crossvalidation)
        num_splits : None or int
            number of splits for kfold
            if None: 10 splits will be validated
        shuffle : bool
            whether or not to shuffle indices for kfold
        random_seed : None or int
            random seed used to seed the kfold (if shuffle is true),
            pytorch and numpy
        label_key : str (default: "label")
            the key to extract the label for stratification from each data
            sample
        train_kwargs : dict
            keyword arguments to specify training behavior
        test_kwargs : dict
            keyword arguments to specify testing behavior
        **kwargs :
            additional keyword arguments (completely passed to self.run())

        See Also
        --------
        :method:`BaseDataManager.update_state_from_dict` for valid keys in
            ``train_kwargs`` and ``test_kwargs``

        """

        metrics_val = {}
        outputs = {}

        metrics = self.params.nested_get("val_metrics", {})

        if num_splits is None:
            num_splits = 10
            logger.warning("num_splits not defined, using default value of \
                                10 splits instead ")

        if isinstance(data.dataset, BaseLazyDataset):
            logger.warning("A lazy dataset is given for stratified kfold. \
                            Iterating over the dataset to extract labels for \
                            stratification may be a massive overhead")

        split_idxs = list(range(len(data.dataset)))
        split_labels = [data.dataset[_idx][label_key] for _idx in split_idxs]

        fold = StratifiedKFold(n_splits=num_splits, shuffle=shuffle,
                               random_state=random_seed)

        for idx, (_train_idxs, test_idxs) in enumerate(fold.split(split_idxs,
                                                                  split_labels)):
            # extract data from single manager
            _train_data = data.get_subset(_train_idxs)
            _split_idxs = list(range(len(_train_data.dataset)))
            _split_labels = [_train_data.dataset[_idx][label_key]
                             for _idx in _split_idxs]

            val_fold = StratifiedShuffleSplit(n_splits=1,
                                              test_size=split_val,
                                              random_state=random_seed)

            for train_idxs, val_idx in val_fold.split(_split_idxs, _split_labels):
                train_data = _train_data.get_subset(train_idxs)
                val_data = _train_data.get_subset(val_idx)

            test_data = data.get_subset(test_idxs)

            # update manager behavior for train and test case
            train_data.update_state_from_dict(train_kwargs)
            val_data.update_state_from_dict(test_kwargs)
            test_data.update_state_from_dict(test_kwargs)
            model = self.run(train_data, val_data,
                             num_epochs=num_epochs,
                             fold=idx,
                             **kwargs)

            _outputs, _metrics_val = self.test(
                self.params, model, test_data, metrics=metrics)

            outputs[str(idx)] = _outputs
            metrics_val[str(idx)] = _metrics_val

        return outputs, metrics_val

    def __str__(self):
        """
        Converts :class:`AbstractExperiment` to string representation

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
        Call :meth:`AbstractExperiment.run`

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

    @abstractmethod
    def save(self):
        """
        Saves the Whole experiments

        Raises
        ------
        NotImplementedError
            If not overwritten in subclass

        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def load(file_name):
        """
        Loads whole experiment

        Parameters
        ----------
        file_name : str
            file_name to load the experiment from

        Raises
        ------
        NotImplementedError
            if not overwritten in subclass

        """
        raise NotImplementedError()


if "TORCH" in get_backends():

    import torch
    from .train_utils import create_optims_default_pytorch, \
        convert_torch_tensor_to_npy
    from .pytorch_trainer import PyTorchNetworkTrainer as PTNetworkTrainer
    from ..models import AbstractPyTorchNetwork

    class PyTorchExperiment(AbstractExperiment):
        """
        Single Experiment for PyTorch Backend

        See Also
        --------
        :class:`AbstractExperiment`

        """

        def __init__(self,
                     params: Parameters,
                     model_cls: AbstractPyTorchNetwork,
                     name=None,
                     save_path=None,
                     val_score_key=None,
                     optim_builder=create_optims_default_pytorch,
                     checkpoint_freq=1,
                     trainer_cls=PTNetworkTrainer,
                     **kwargs
                     ):
            """

            Parameters
            ----------
            params : :class:`Parameters`
                the training and model parameters
            model_cls :
                the class to instantiate models
            name : str
                the experiment's name,
                default: None -> "UnnamedExperiment"
            save_path : str
                the path to save the experiment to
                (a date-time signature will be appended),
                default: None -> Use current working dir
            val_score_key : str or None
                key to access the metric to monitor for model
                selection and callbacks (often starts with "val_")
            optim_builder : function
                function returning a dictionary of optimizers
                defaults to :function:`create_optims_default_pytorch`
            checkpoint_freq : int
                save checkpoint after each n epochs
                (if set to 1, checkpoints will be saved after each epoch,
                if set to 2, checkpoints will be saved after each
                2 epochs etc.)
            trainer_cls :
                class defining the actual trainer,
                defaults to :class:`PyTorchNetworkTrainer`,
                which should be suitable for most cases,
                but can easily be overwritten and exchanged if necessary
            **kwargs :
                additional keyword arguments

            """

            if isinstance(params, str):
                with open(params, "rb") as f:
                    params = pickle.load(f)

            n_epochs = params.nested_get("num_epochs")
            AbstractExperiment.__init__(self, n_epochs)

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

            if val_score_key is None:

                if params.nested_get("val_metrics", False):
                    val_score_key = sorted(
                        params.nested_get("val_metrics").keys())[0]

            self.val_score_key = val_score_key

            self.params = params
            self.model_cls = model_cls
            self.kwargs = kwargs
            self._optim_builder = optim_builder
            self.checkpoint_freq = checkpoint_freq
            self._run = 0

            # log HyperParameters
            logger.info({"text": {"text":
                                  str(params) + "\n\tmodel_class = %s"
                                  % model_cls.__class__.__name__}})

        def setup(self, params: Parameters, **kwargs):
            """
            Perform setup of Network Trainer

            Parameters
            ----------
            params : :class:`Parameters`
                the parameters to construct a model and network trainer
            **kwargs :
                keyword arguments

            """
            model_params = params.permute_training_on_top().model

            model_kwargs = {**model_params.fixed, **model_params.variable}

            model = self.model_cls(**model_kwargs)

            training_params = params.permute_training_on_top().training
            losses = training_params.nested_get("losses")
            optimizer_cls = training_params.nested_get("optimizer_cls")
            optimizer_params = training_params.nested_get("optimizer_params")
            train_metrics = training_params.nested_get("train_metrics", {})
            lr_scheduler_cls = training_params.nested_get("lr_sched_cls", None)
            lr_scheduler_params = training_params.nested_get("lr_sched_params",
                                                             {})
            val_metrics = training_params.nested_get("val_metrics", {})

            return self.trainer_cls(
                network=model,
                save_path=os.path.join(
                    self.save_path,
                    "checkpoints",
                    "run_%02d" % self._run),
                losses=losses,
                optimizer_cls=optimizer_cls,
                optimizer_params=optimizer_params,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                lr_scheduler_cls=lr_scheduler_cls,
                lr_scheduler_params=lr_scheduler_params,
                optim_fn=self._optim_builder,
                save_freq=self.checkpoint_freq,
                **self.kwargs,
                **kwargs
            )

        def run(self,
                train_data: BaseDataManager,
                val_data: typing.Union[BaseDataManager, None],
                params: typing.Optional[Parameters] = None,
                **kwargs):
            """
            trains single model

            Parameters
            ----------
            train_data : BaseDataManager
                holds the trainset
            val_data : BaseDataManager or None
                holds the validation set (if None: Model will not be validated)
            params : :class:`Parameters`
                the parameters to construct a model and network trainer
            **kwargs :
                holds additional keyword arguments
                (which are completly passed to the trainers init)

            Returns
            -------
            :class:`BaseNetworkTrainer`
                trainer of trained network

            Raises
            ------
            ValueError
                Class has no Attribute ``params`` and no parameters were given as
                function argument

            """

            if params is None:
                if hasattr(self, "params"):
                    params = self.params
                else:
                    raise ValueError("No parameters given")

            else:
                self.params = params

            training_params = params.permute_training_on_top().training

            trainer = self.setup(params, **kwargs)
            self._run += 1
            num_epochs = kwargs.get("num_epochs", training_params.nested_get(
                "num_epochs"))
            return trainer.train(num_epochs, train_data, val_data,
                                 self.val_score_key,
                                 self.kwargs.get("val_score_mode",
                                                 "lowest")
                                 )

        def test(self,
                 params: Parameters,
                 network: AbstractPyTorchNetwork,
                 datamgr_test: BaseDataManager,
                 metrics: dict,
                 prepare_batch=None,
                 verbose=False,
                 ** kwargs):
            """
            Executes prediction for all items in datamgr_test with network

            Parameters
            ----------
            params : :class:`Parameters`
                the parameters to construct a model
            network : :class:'AbstractPyTorchNetwork'
                the network to train
            datamgr_test : :class:'BaseDataManager'
                holds the test data
            **kwargs :
                holds additional keyword arguments
                (which are completly passed to the trainers init)

            Returns
            -------
            np.ndarray
                predictions from batches
            list of np.ndarray
                labels from batches
            dict
                dictionary containing the mean validation metrics and
                the mean loss values
            """

            if prepare_batch is None:
                prepare_batch = partial(network.prepare_batch,
                                        input_device=torch.device("cpu"),
                                        output_device=torch.device("cpu"))

            return super().test(params, network, datamgr_test, metrics,
                                prepare_batch, convert_torch_tensor_to_npy, verbose,
                                **kwargs)

        def save(self):
            """
            Saves the Whole experiments

            """
            with open(os.path.join(self.save_path, "experiment.delira.pkl"),
                      "wb") as f:
                pickle.dump(self, f)

            self.params.save(os.path.join(self.save_path, "parameters"))

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

        def __getstate__(self):
            return vars(self)

        def __setstate__(self, state):
            vars(self).update(state)

        def kfold(self, num_epochs: int,
                  data: typing.Union[typing.List[BaseDataManager],
                                     BaseDataManager],
                  num_splits=None, shuffle=False, random_seed=None,
                  train_kwargs={}, test_kwargs={}, **kwargs):

            if random_seed is not None:
                torch.manual_seed(random_seed)

            super().kfold(num_epochs, data, num_splits, shuffle, random_seed,
                          train_kwargs, test_kwargs, **kwargs)

        def stratified_kfold(self, num_epochs: int,
                             data: BaseDataManager,
                             num_splits=None, shuffle=False, random_seed=None,
                             label_key="label", train_kwargs={}, test_kwargs={},
                             **kwargs):

            if random_seed is not None:
                torch.manual_seed(random_seed)

            super().stratified_kfold(num_epochs, data, num_splits, shuffle,
                                     random_seed, label_key, train_kwargs,
                                     test_kwargs, **kwargs)

if "TF" in get_backends():

    from .tf_trainer import TfNetworkTrainer
    from .train_utils import create_optims_default_tf
    from ..models import AbstractTfNetwork
    from .parameters import Parameters
    import tensorflow as tf

    class TfExperiment(AbstractExperiment):
        """
        Single Experiment for Tf Backend

        See Also
        --------
        :class:`AbstractExperiment`

        """

        def __init__(self,
                     params: typing.Union[Parameters, str],
                     model_cls: AbstractTfNetwork,
                     name=None,
                     save_path=None,
                     val_score_key=None,
                     optim_builder=create_optims_default_tf,
                     checkpoint_freq=1,
                     trainer_cls=TfNetworkTrainer,
                     **kwargs
                     ):

            if isinstance(params, str):
                with open(params, "rb") as f:
                    params = pickle.load(f)

            n_epochs = params.nested_get("num_epochs")
            AbstractExperiment.__init__(self, n_epochs)

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

            if val_score_key is None:
                if params.nested_get("val_metrics", False):
                    val_score_key = sorted(
                        params.nested_get("val_metrics").keys())[0]
            self.val_score_key = val_score_key

            self.params = params
            self.model_cls = model_cls
            self.kwargs = kwargs
            self._optim_builder = optim_builder
            self.checkpoint_freq = checkpoint_freq
            self._run = 0

            # log HyperParameters
            logger.info({"text": {"text":
                                  str(params) + "\n\tmodel_class = %s"
                                  % model_cls.__class__.__name__}})

        def setup(self, params: Parameters, **kwargs):
            """
            Perform setup of Network Trainer

            Parameters
            ----------
            params : :class:`Parameters`
                the parameters to construct a model and network trainer
            **kwargs :
                keyword arguments

            """

            model_params = params.permute_training_on_top().model

            model_kwargs = {**model_params.fixed, **model_params.variable}

            tf.reset_default_graph()

            model = self.model_cls(**model_kwargs)

            training_params = params.permute_training_on_top().training
            losses = training_params.nested_get("losses")
            optimizer_cls = training_params.nested_get("optimizer_cls")
            optimizer_params = training_params.nested_get("optimizer_params")
            train_metrics = training_params.nested_get("train_metrics", {})
            lr_scheduler_cls = training_params.nested_get("lr_sched_cls", None)
            lr_scheduler_params = training_params.nested_get("lr_sched_params",
                                                             {})
            val_metrics = training_params.nested_get("val_metrics", {})

            return self.trainer_cls(
                network=model,
                save_path=os.path.join(
                    self.save_path,
                    "checkpoints",
                    "run_%02d" % self._run),
                losses=losses,
                optimizer_cls=optimizer_cls,
                optimizer_params=optimizer_params,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                lr_scheduler_cls=lr_scheduler_cls,
                lr_scheduler_params=lr_scheduler_params,
                optim_fn=self._optim_builder,
                save_freq=self.checkpoint_freq,
                **self.kwargs,
                **kwargs
            )

        def run(self,
                train_data: BaseDataManager,
                val_data: typing.Union[BaseDataManager, None],
                params: typing.Optional[Parameters] = None,
                **kwargs):
            """
            trains single model

            Parameters
            ----------
            train_data : BaseDataManager
                holds the trainset
            val_data : BaseDataManager or None
                holds the validation set (if None: Model will not be validated)
            params : :class:`Parameters`
                the parameters to construct a model and network trainer
            **kwargs :
                holds additional keyword arguments
                (which are completly passed to the trainers init)

            Returns
            -------
            :class:`BaseNetworkTrainer`
                trainer of trained network

            Raises
            ------
            ValueError
                Class has no Attribute ``params`` and no parameters were given as
                function argument
            """

            if params is None:
                if hasattr(self, "params"):
                    params = self.params
                else:
                    raise ValueError("No parameters given")

            else:
                self.params = params

            training_params = params.permute_training_on_top().training

            trainer = self.setup(params, **kwargs)
            self._run += 1
            num_epochs = kwargs.get("num_epochs", training_params.nested_get(
                "num_epochs"))
            return trainer.train(num_epochs, train_data, val_data,
                                 self.val_score_key,
                                 self.kwargs.get("val_score_mode",
                                                 "lowest")
                                 )

        def save(self):
            """
            Saves the Whole experiments

            """
            with open(os.path.join(self.save_path, "experiment.delira.pkl"),
                      "wb") as f:
                pickle.dump(self, f)

                self.params.save(os.path.join(self.save_path, "parameters"))

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

        def __getstate__(self):
            return vars(self)

        def __setstate__(self, state):
            vars(self).update(state)

        def kfold(self, num_epochs: int,
                  data: typing.Union[typing.List[BaseDataManager],
                                     BaseDataManager],
                  num_splits=None, shuffle=False, random_seed=None,
                  train_kwargs={}, test_kwargs={}, **kwargs):

            if random_seed is not None:
                tf.set_random_seed(random_seed)

            super().kfold(num_epochs, data, num_splits, shuffle, random_seed,
                          train_kwargs, test_kwargs, **kwargs)

        def stratified_kfold(self, num_epochs: int,
                             data: BaseDataManager,
                             num_splits=None, shuffle=False, random_seed=None,
                             label_key="label", train_kwargs={}, test_kwargs={},
                             **kwargs):

            if random_seed is not None:
                tf.set_random_seed(random_seed)

            super().stratified_kfold(num_epochs, data, num_splits, shuffle,
                                     random_seed, label_key, train_kwargs,
                                     test_kwargs, **kwargs)

        def stratified_kfold_predict(self, num_epochs: int,
                                     data: BaseDataManager,
                                     split_val=0.2,
                                     num_splits=None,
                                     shuffle=False,
                                     random_seed=None,
                                     label_key="label",
                                     train_kwargs={}, test_kwargs={},
                                     **kwargs):

            if random_seed is not None:
                tf.set_random_seed(random_seed)

            return super().stratified_kfold_predict(num_epochs, data, split_val,
                                                    num_splits, shuffle,
                                                    random_seed, label_key,
                                                    train_kwargs,
                                                    test_kwargs, **kwargs)

        def test(self,
                 params: Parameters,
                 network: AbstractNetwork,
                 datamgr_test: BaseDataManager,
                 metrics: dict,
                 verbose=False,             
                 **kwargs):

            """
            Executes prediction for all items in datamgr_test with network

            Parameters
            ----------
            params : :class:`Parameters`
                the parameters to construct a model
            network : :class:'AbstractPyTorchNetwork'
                the network to train
            datamgr_test : :class:'BaseDataManager'
                holds the test data
            **kwargs :
                holds additional keyword arguments
                (which are completly passed to the trainers init)

            Returns
            -------
            np.ndarray
                predictions from batches
            list of np.ndarray
                labels from batches
            dict
                dictionary containing the mean validation metrics and
                the mean loss values
            """
            return super().test(params, network, datamgr_test, metrics,
                                verbose=verbose, **kwargs)