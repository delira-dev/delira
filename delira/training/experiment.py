

from ..utils import now
from ..data_loading import BaseDataManager, ConcatDataManager
from .. import __version__ as delira_version
from .parameters import Parameters
from trixi.experiment import Experiment as TrixiExperiment
import os
import logging
import yaml
import typing
import numpy as np

import pickle

from abc import abstractmethod
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from datetime import datetime
from inspect import signature
from functools import partial

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
    def run(self, train_data: typing.Union[BaseDataManager, ConcatDataManager],
            val_data: typing.Optional[typing.Union[BaseDataManager,
                                                   ConcatDataManager]] = None,
            params: typing.Optional[Parameters] = None,
            **kwargs):
        """
        trains single model

        Parameters
        ----------
        train_data : :class:`BaseDataManager` or :class:`ConcatDataManager`
            data manager containing the training data
        val_data : :class:`BaseDataManager` or :class:`ConcatDataManager`
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


try:

    import torch
    from .train_utils import create_optims_default_pytorch
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
            
            if val_score_key is None and params.nested_get("metrics"):
                val_score_key = sorted(params.nested_get("metrics").keys())[0]

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

            model_kwargs = {}
            for key in signature(self.model_cls.__init__).parameters.keys():
                if key in ["self", "args", "kwargs"]:
                    continue
                try:
                    model_kwargs[key] = model_params.nested_get(key)

                except KeyError:
                    pass

            model = self.model_cls(**model_kwargs)

            training_params = params.permute_training_on_top().training
            criterions = training_params.nested_get("criterions")
            optimizer_cls = training_params.nested_get("optimizer_cls")
            optimizer_params = training_params.nested_get("optimizer_params")
            metrics = training_params.nested_get("metrics")
            lr_scheduler_cls = training_params.nested_get("lr_sched_cls")
            lr_scheduler_params = training_params.nested_get("lr_sched_params")
            return self.trainer_cls(
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
                params: typing.Optional[Parameters] = None,
                **kwargs):
            """
            trains single model

            Parameters
            ----------
            train_data : BaseDataManager or ConcatDataManager
                holds the trainset
            val_data : BaseDataManager or ConcatDataManager or None
                holds the validation set (if None: Model will not be validated)
            params : :class:`Parameters`
                the parameters to construct a model and network trainer
            **kwargs :
                holds additional keyword arguments 
                (which are completly passed to the trainers init)

            Returns
            -------
            :class:`AbstractNetworkTrainer`
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

except ImportError as e:
    raise e
