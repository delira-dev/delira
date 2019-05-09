import typing
import logging
import pickle
import os
from datetime import datetime
from functools import partial

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, \
    StratifiedShuffleSplit, ShuffleSplit
from trixi.experiment import Experiment as TrixiExperiment

from delira import get_backends

from ..data_loading import BaseDataManager
from ..models import AbstractNetwork

from .parameters import Parameters
from .base_trainer import BaseNetworkTrainer
from .predictor import Predictor

logger = logging.getLogger(__name__)


class BaseExperiment(TrixiExperiment):

    def __init__(self,
                 params: Parameters,
                 model_cls: AbstractNetwork,
                 n_epochs=None,
                 name=None,
                 save_path=None,
                 key_mapping=None,
                 val_score_key=None,
                 optim_builder=None,
                 checkpoint_freq=1,
                 trainer_cls=BaseNetworkTrainer,
                 **kwargs):

        if n_epochs is None:
            n_epochs = params.nested_get("n_epochs",
                                         params.nested_get("num_epochs"))

        super().__init__(n_epochs)

        # params could also be a file containing a pickled instance of parameters
        if isinstance(params, str):
            with open(params, "rb") as f:
                params = pickle.load(f)

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

        assert key_mapping is not None
        self.key_mapping = key_mapping

        self.params = params
        self.model_cls = model_cls

        self._optim_builder = optim_builder
        self.checkpoint_freq = checkpoint_freq

        self._run = 0

        self.kwargs = kwargs

    def setup(self, params, training=True, **kwargs):
        if training:
            return self._setup_training(params, **kwargs)

        return self._setup_test(params, **kwargs)

    def _setup_training(self, params, **kwargs):
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
            val_metrics=val_metrics,
            lr_scheduler_cls=lr_scheduler_cls,
            lr_scheduler_params=lr_scheduler_params,
            optim_fn=self._optim_builder,
            save_freq=self.checkpoint_freq,
            **kwargs
        )

    def _setup_test(self, params, model, convert_batch_to_npy_fn,
                    prepare_batch_fn, **kwargs):
        predictor = Predictor(model=model, key_mapping=self.key_mapping,
                              convert_batch_to_npy_fn=convert_batch_to_npy_fn,
                              prepare_batch_fn=prepare_batch_fn, **kwargs)
        return predictor

    def run(self, train_data: BaseDataManager,
            val_data: BaseDataManager = None,
            params: Parameters = None, **kwargs):

        params = self._resolve_params(params)
        kwargs = self._resolve_kwargs(kwargs)

        training_params = params.training

        trainer = self.setup(params, training=True, **kwargs)

        self._run += 1

        num_epochs = kwargs.get("num_epochs", training_params.nested_get(
            "num_epochs", self.n_epochs))

        if num_epochs is None:
            num_epochs = self.n_epochs

        return trainer.train(num_epochs, train_data, val_data,
                             self.val_score_key, kwargs.get("val_score_mode",
                                                            "lowest"))

    def resume(self, save_path: str, train_data: BaseDataManager,
               val_data: BaseDataManager = None,
               params: Parameters = None, **kwargs):
        return self.run(train_data=train_data, val_data=val_data, params=params,
                        save_path=save_path, **kwargs)

    def test(self, network, test_data: BaseDataManager, params,
             metrics: dict, metric_keys=None,
             verbose=False, prepare_batch=lambda x: x,
             convert_fn=lambda x: x, **kwargs):

        params = self._resolve_params(params)
        kwargs = self._resolve_kwargs(kwargs)

        predictor = self.setup(params, training=False, model=network,
                               convert_batch_to_npy_fn=convert_fn,
                               prepare_batch_fn=prepare_batch, **kwargs)

        return predictor.predict_data_mgr(test_data, 1, metrics,
                                          metric_keys, verbose)

    def kfold(self, data: BaseDataManager, metrics: dict, num_epochs=None,
              num_splits=None, shuffle=False, random_seed=None,
              split_type="random", val_split=0.2, label_key="label",
              train_kwargs: dict = None, metric_keys: dict = None,
              test_kwargs: dict = None, params=None, verbose=False, **kwargs):

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
            split_labels = [data.dataset[_idx][label_key] for _idx in split_idxs]
        else:
            raise ValueError("split_type must be one of "
                             "['random', 'stratified'], but got: %s"
                             % str(split_type))

        fold = split_cls(n_splits=num_splits, shuffle=shuffle,
                         random_state=random_seed)

        if random_seed is not None:
            np.random.seed(random_seed)

        # iterate over folds
        for idx, (train_idxs, test_idxs) in enumerate(fold.split(split_idxs,
                                                                 split_labels)):

            # extract data from single manager
            train_data = data.get_subset(train_idxs)
            test_data = data.get_subset(test_idxs)

            train_data.update_state_from_dict(train_kwargs)
            test_data.update_state_from_dict(test_kwargs)

            val_data = None
            if val_split is not None:
                if split_type == "random":
                    # split_labels are ignored for random splitting, set them to
                    # split_idxs just ensures same length
                    train_labels = train_idxs
                elif split_type == "stratified":
                    # iterate over dataset to get labels for stratified splitting
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
                    val_data.update_state_from_dict(test_kwargs)

                    train_data = train_data.get_subset(_train_idxs)

            model = self.run(train_data=train_data, val_data=val_data,
                             params=params, num_epochs=num_epochs, fold=idx,
                             **kwargs)

            _outputs, _metrics_test = self.test(model, test_data, params,
                                                metrics=metrics,
                                                metric_keys=metric_keys,
                                                verbose=verbose)

            outputs[str(idx)] = _outputs
            metrics_test[str(idx)] = _metrics_test

        return outputs, metrics_test

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

    def _resolve_params(self, params: typing.Union[dict, None]):
        if params is None:
            params = Parameters()

        elif not isinstance(params, Parameters):
            _params = params
            params = Parameters()
            params.update(_params)

        if hasattr(self, "params"):
            params = params.permute_training_on_top()
            params.update(self.params.permute_training_on_top())

        return params

    def _resolve_kwargs(self, kwargs: typing.Union[dict, None]):

        if kwargs is None:
            kwargs = {}

        if hasattr(self, "kwargs"):
            _kwargs = kwargs
            kwargs = self.kwargs
            kwargs.update(_kwargs)

        return kwargs

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        vars(self).update(state)


if "TORCH" in get_backends():

    from .train_utils import create_optims_default_pytorch, \
        convert_torch_tensor_to_npy
    from .pytorch_trainer import PyTorchNetworkTrainer as PTNetworkTrainer
    from ..models import AbstractPyTorchNetwork
    import torch


    class PyTorchExperiment(BaseExperiment):
        def __init__(self,
                     params: Parameters,
                     model_cls: AbstractPyTorchNetwork,
                     n_epochs=None,
                     name=None,
                     save_path=None,
                     key_mapping=None,
                     val_score_key=None,
                     optim_builder=create_optims_default_pytorch,
                     checkpoint_freq=1,
                     trainer_cls=PTNetworkTrainer,
                     **kwargs):

            if key_mapping is None:
                key_mapping = {"x": "data"}
            super().__init__(params=params, model_cls=model_cls,
                             n_epochs=n_epochs, name=name, save_path=save_path,
                             key_mapping=key_mapping,
                             val_score_key=val_score_key,
                             optim_builder=optim_builder,
                             checkpoint_freq=checkpoint_freq,
                             trainer_cls=trainer_cls,
                             **kwargs)

        def kfold(self, data: BaseDataManager, metrics: dict, num_epochs=None,
                  num_splits=None, shuffle=False, random_seed=None,
                  split_type="random", val_split=0.2, label_key="label",
                  train_kwargs: dict = None, test_kwargs: dict = None,
                  metric_keys: dict = None, params=None, verbose=False,
                  **kwargs):

            if random_seed is not None:
                torch.manual_seed(random_seed)

            return super().kfold(data=data, metrics=metrics,
                                 num_epochs=num_epochs,
                                 num_splits=num_splits, shuffle=shuffle,
                                 random_seed=random_seed, split_type=split_type,
                                 val_split=val_split, label_key=label_key,
                                 train_kwargs=train_kwargs,
                                 test_kwargs=test_kwargs,
                                 metric_keys=metric_keys, params=params,
                                 verbose=verbose, **kwargs)

        def test(self, network, test_data: BaseDataManager, params,
                 metrics: dict, metric_keys=None,
                 verbose=False, prepare_batch=None,
                 convert_fn=None, **kwargs):

            if prepare_batch is None:
                prepare_batch = partial(network.prepare_batch,
                                        input_device=torch.device("cpu"),
                                        output_device=torch.device("cpu"))

            if convert_fn is None:
                convert_fn = convert_torch_tensor_to_npy

            return super().test(network=network, test_data=test_data,
                                params=params,
                                metrics=metrics, metric_keys=metric_keys,
                                verbose=verbose, prepare_batch=prepare_batch,
                                convert_fn=convert_fn, **kwargs)

if "TF" in get_backends():
    from .tf_trainer import TfNetworkTrainer
    from .train_utils import create_optims_default_tf, convert_tf_tensor_to_npy
    from ..models import AbstractTfNetwork
    from .parameters import Parameters
    import tensorflow as tf

    class TfExperiment(BaseExperiment):
        def __init__(self,
                     params: Parameters,
                     model_cls: AbstractTfNetwork,
                     n_epochs=None,
                     name=None,
                     save_path=None,
                     key_mapping=None,
                     val_score_key=None,
                     optim_builder=create_optims_default_tf,
                     checkpoint_freq=1,
                     trainer_cls=TfNetworkTrainer,
                     **kwargs):

            if key_mapping is None:
                key_mapping = {"images": "data"}
            super().__init__(params=params, model_cls=model_cls,
                             n_epochs=n_epochs, name=name, save_path=save_path,
                             key_mapping=key_mapping,
                             val_score_key=val_score_key,
                             optim_builder=optim_builder,
                             checkpoint_freq=checkpoint_freq,
                             trainer_cls=trainer_cls,
                             **kwargs)

        def setup(self, params, training=True, **kwargs):

            tf.reset_default_graph()

            return super().setup(params=params, training=training, **kwargs)

        def kfold(self, data: BaseDataManager, metrics: dict, num_epochs=None,
                  num_splits=None, shuffle=False, random_seed=None,
                  split_type="random", val_split=0.2, label_key="label",
                  train_kwargs: dict = None, test_kwargs: dict = None,
                  metric_keys: dict = None, params=None, verbose=False,
                  **kwargs):

            if random_seed is not None:
                tf.set_random_seed(random_seed)

            return super().kfold(data=data, metrics=metrics,
                                 num_epochs=num_epochs,
                                 num_splits=num_splits, shuffle=shuffle,
                                 random_seed=random_seed, split_type=split_type,
                                 val_split=val_split, label_key=label_key,
                                 train_kwargs=train_kwargs,
                                 test_kwargs=test_kwargs,
                                 metric_keys=metric_keys, params=params,
                                 verbose=verbose, **kwargs)

        def test(self, network, test_data: BaseDataManager, params,
                 metrics: dict, metric_keys=None,
                 verbose=False, prepare_batch=lambda x: x,
                 convert_fn=None, **kwargs):

            if convert_fn is None:
                convert_fn = convert_tf_tensor_to_npy

            return super().test(network=network, test_data=test_data,
                                params=params,
                                metrics=metrics, metric_keys=metric_keys,
                                verbose=verbose, prepare_batch=prepare_batch,
                                convert_fn=convert_fn, **kwargs)
