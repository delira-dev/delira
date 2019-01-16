import os
import logging
import typing
import numpy as np
import torch
import pickle
import tensorflow as tf
from abc import abstractmethod
from sklearn.model_selection import KFold
from datetime import datetime
from trixi.experiment import Experiment as TrixiExperiment
from .hyper_params import Hyperparameters
from ..data_loading import BaseDataManager, ConcatDataManager
from ..models import AbstractPyTorchNetwork
from ..models import AbstractTfNetwork
from .pytorch_trainer import PyTorchNetworkTrainer as PTNetworkTrainer
from .tf_trainer import TfNetworkTrainer as TFNetworkTrainer
from .train_utils import create_optims_default_pytorch, create_optims_default_tf

logger = logging.getLogger(__name__)

NOT_IMPLEMENTED_KEYS = []


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
        self.current_trainer = None

    @abstractmethod
    def setup(self):
        """
        Abstract Method to setup a :class:`AbstractNetworkTrainer`

        Raises
        ------
        NotImplementedError
            if not overwritten by subclass

        """
        raise NotImplementedError()

    @abstractmethod
    def run(self, train_data: typing.Union[BaseDataManager, ConcatDataManager],
            val_data: typing.Union[BaseDataManager, ConcatDataManager, None],
            **kwargs):
        """

        Parameters
        ----------
        train_data : :class:`BaseDataManager` or :class:`ConcatDataManager`
            data manager containing the training data
        val_data : :class:`BaseDataManager` or :class:`ConcatDataManager`
            data manager containing the validation data
        **kwargs :
            keyword arguments

        Raises
        ------
        NotImplementedError
            if not overwritten by subclass

        """

        raise NotImplementedError()

    def kfold(self, num_epochs: int, data: typing.List[BaseDataManager],
              num_splits=None, shuffle=False, random_seed=None, **kwargs):
        """
        Runs K-Fold Crossvalidation

        Parameters
        ----------
        num_epochs : int
            number of epochs to train the model
        data : list of BaseDataManager
            list of datamanagers (will be split for crossvalidation)
        num_splits : None or int
            number of splits for kfold
            if None: len(data) splits will be validated
        shuffle : bool
            whether or not to shuffle indices for kfold
        random_seed : None or int
            random seed used to seed the kfold (if shuffle is true),
            pytorch and numpy
        **kwargs :
            additional keyword arguments (completely passed to self.run())

        """

        if num_splits is None:
            num_splits = len(data)

        fold = KFold(n_splits=num_splits, shuffle=shuffle,
                     random_state=random_seed)

        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        for idx, (train_idxs, test_idxs) in enumerate(fold.split(data)):
            self.run(ConcatDataManager(
                [data[_idx] for _idx in train_idxs]),
                ConcatDataManager([data[_idx] for _idx in test_idxs]),
                num_epochs=num_epochs,
                fold=idx,
                **kwargs)

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
        :class:`AbstractNetworkTrainer`
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


class PyTorchExperiment(AbstractExperiment):
    """
    Single Experiment for PyTorch Backend

    See Also
    --------
    :class:`AbstractExperiment`

    """
    def __init__(self,
                 hyper_params: typing.Union[Hyperparameters, str],
                 model_cls: AbstractPyTorchNetwork,
                 model_kwargs: dict,
                 name=None,
                 save_path=None,
                 val_score_key=None,
                 optim_builder=create_optims_default_pytorch,
                 checkpoint_freq=1,
                 **kwargs
                 ):

        if isinstance(hyper_params, str):
            hyper_params = Hyperparameters.from_file(hyper_params)

        n_epochs = hyper_params.num_epochs
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

        if val_score_key is None and hyper_params.metrics:
            val_score_key = sorted(hyper_params.metrics.keys())[0]

        self.val_score_key = val_score_key

        self.hyper_params = hyper_params
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs
        self.kwargs = kwargs
        self._optim_builder = optim_builder
        self.checkpoint_freq = checkpoint_freq
        self._run = 0

        # log HyperParameters
        logger.info({"text": {"text":
                                  str(hyper_params) + "\n\tmodel_class = %s"
                                  % model_cls.__class__.__name__}})

    def setup(self, **kwargs):
        """
        Perform setup of Network Trainer

        Parameters
        ----------
        **kwargs :
            keyword arguments

        """
        model = self.model_cls(**self.model_kwargs)

        criterions = self.hyper_params.criterions
        optimizer_cls = self.hyper_params.optimizer_cls
        optimizer_params = self.hyper_params.optimizer_params
        metrics = self.hyper_params.metrics
        lr_scheduler_cls = self.hyper_params.lr_sched_cls
        lr_scheduler_params = self.hyper_params.lr_sched_params
        self.current_trainer = PTNetworkTrainer(
            network=model,
            save_path=os.path.join(
                self.save_path,
                "checkpoints",
                "run_%02d" % self._run),
            criterions=criterions,
            optimizer_cls=optimizer_cls,
            optimizer_params=optimizer_params,
            metrics=metrics,
            lr_scheduler_cls=lr_scheduler_cls,
            lr_scheduler_params=lr_scheduler_params,
            optim_fn=self._optim_builder,
            save_freq=self.checkpoint_freq,
            **self.kwargs,
            **kwargs
        )

    def run(self,
            train_data: typing.Union[BaseDataManager, ConcatDataManager],
            val_data: typing.Union[BaseDataManager, ConcatDataManager, None],
            **kwargs):
        """
        trains single model

        Parameters
        ----------
        train_data : BaseDataManager or ConcatDataManager
            holds the trainset
        val_data : BaseDataManager or ConcatDataManager or None
            holds the validation set (if None: Model will not be validated)
        **kwargs :
            holds additional keyword arguments 
            (which are completly passed to the trainers init)

        Returns
        -------
        :class:`AbstractNetworkTrainer`
            trainer of trained network

        """
        self.setup(**kwargs)
        self._run += 1
        num_epochs = kwargs.get("num_epochs", self.hyper_params.num_epochs)
        return self.current_trainer.train(num_epochs, train_data, val_data,
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

        self.hyper_params.export_to_files(self.save_path, True)

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

class TfExperiment(AbstractExperiment):
    """
    Single Experiment for Tf Backend

    See Also
    --------
    :class:`AbstractExperiment`

    """
    def __init__(self,
                 hyper_params: typing.Union[Hyperparameters, str],
                 model_cls: AbstractTfNetwork,
                 model_kwargs: dict,
                 name=None,
                 save_path=None,
                 val_score_key=None,
                 optim_builder=create_optims_default_tf,
                 checkpoint_freq=1,
                 **kwargs
                 ):

        if isinstance(hyper_params, str):
            hyper_params = Hyperparameters.from_file(hyper_params)

        n_epochs = hyper_params.num_epochs
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

        if val_score_key is None and hyper_params.metrics:
            val_score_key = sorted(hyper_params.metrics.keys())[0]

        self.val_score_key = val_score_key

        self.hyper_params = hyper_params
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs
        self.kwargs = kwargs
        self._optim_builder = optim_builder
        self.checkpoint_freq = checkpoint_freq
        self._run = 0

        # log HyperParameters
        logger.info({"text": {"text":
                                  str(hyper_params) + "\n\tmodel_class = %s"
                                  % model_cls.__class__.__name__}})

    def setup(self, **kwargs):
        """
        Perform setup of Network Trainer

        Parameters
        ----------
        **kwargs :
            keyword arguments

        """

        tf.reset_default_graph()

        criterions = self.hyper_params.criterions
        optimizer_cls = self.hyper_params.optimizer_cls
        optimizer_params = self.hyper_params.optimizer_params

        model = self.model_cls(**self.model_kwargs)
        metrics = self.hyper_params.metrics
        lr_scheduler_cls = self.hyper_params.lr_sched_cls
        lr_scheduler_params = self.hyper_params.lr_sched_params
        self.current_trainer = TFNetworkTrainer(
            network=model,
            save_path=os.path.join(
                self.save_path,
                "checkpoints",
                "run_%02d" % self._run),
            losses=criterions,
            optimizer_cls=optimizer_cls,
            optimizer_params=optimizer_params,
            metrics=metrics,
            lr_scheduler_cls=lr_scheduler_cls,
            lr_scheduler_params=lr_scheduler_params,
            optim_fn=self._optim_builder,
            save_freq=self.checkpoint_freq,
            **self.kwargs,
            **kwargs
        )

    def run(self,
            train_data: typing.Union[BaseDataManager, ConcatDataManager],
            val_data: typing.Union[BaseDataManager, ConcatDataManager, None],
            **kwargs):
        """
        trains single model

        Parameters
        ----------
        train_data : BaseDataManager or ConcatDataManager
            holds the trainset
        val_data : BaseDataManager or ConcatDataManager or None
            holds the validation set (if None: Model will not be validated)
        **kwargs :
            holds additional keyword arguments
            (which are completly passed to the trainers init)

        Returns
        -------
        :class:`AbstractNetworkTrainer`
            trainer of trained network

        """

        self.setup(**kwargs)
        self._run += 1
        num_epochs = kwargs.get("num_epochs", self.hyper_params.num_epochs)
        return self.current_trainer.train(num_epochs, train_data, val_data,
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

        self.hyper_params.export_to_files(self.save_path, True)

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
