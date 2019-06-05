import os
import logging
import numpy as np
from tqdm.auto import tqdm
from ..data_loading.sampler import RandomSampler, RandomSamplerNoReplacement
from .base_trainer import BaseNetworkTrainer
from ..data_loading import BaseDataManager
from functools import partial

from delira import get_backends

logger = logging.getLogger(__name__)

if "SKLEARN" in get_backends():
    from .train_utils import convert_batch_to_numpy_identity
    from .train_utils import create_empty_optim_dict as create_optims_default
    from ..io.sklearn import save_checkpoint, load_checkpoint
    from ..models import SklearnEstimator

    class SklearnEstimatorTrainer(BaseNetworkTrainer):
        """
        Train and Validate a ``sklearn`` estimator

        See Also
        --------
        :class:`SkLearnEstimator`

        """

        def __init__(self,
                     estimator: SklearnEstimator,
                     save_path: str,
                     key_mapping,
                     train_metrics={},
                     val_metrics={},
                     save_freq=1,
                     logging_type="tensorboardx",
                     logging_kwargs={},
                     fold=0,
                     callbacks=[],
                     start_epoch=1,
                     metric_keys=None,
                     convert_batch_to_npy_fn=convert_batch_to_numpy_identity,
                     val_freq=1,
                     ** kwargs):
            """

            Parameters
            ----------
            estimator : :class:`SklearnEstimator`
                the estimator to train
            save_path : str
                path to save networks to
            key_mapping : dict
                a dictionary containing the mapping from the ``data_dict`` to
                the actual model's inputs.
                E.g. if a model accepts one input named 'x' and the data_dict
                contains one entry named 'data' this argument would have to
                be ``{'x': 'data'}``
            train_metrics : dict, optional
                metrics, which will be evaluated during train phase
                (should work on framework's tensor types)
            val_metrics : dict, optional
                metrics, which will be evaluated during test phase
                (should work on numpy arrays)
            save_freq : int
                integer specifying how often to save the current model's state.
                State is saved every state_freq epochs
            logging_type : str or callable
                the type of logging. If string: it must be one of
                ["visdom", "tensorboardx"]
                If callable: it must be a logging handler class
            logging_kwargs : dict
                dictionary containing all logging keyword arguments
            fold : int
                current cross validation fold (0 per default)
            callbacks : list
                initial callbacks to register
            start_epoch : int
                epoch to start training at
            metric_keys : dict
                dict specifying which batch_dict entry to use for which metric as
                target; default: None, which will result in key "label" for all
                metrics
            convert_batch_to_npy_fn : type, optional
                function converting a batch-tensor to numpy, per default this is
                a function, returning the inputs without changing anything
            val_freq : int
                validation frequency specifying how often to validate the trained
                model (a value of 1 denotes validating every epoch,
                a value of 2 denotes validating every second epoch etc.);
                defaults to 1
            **kwargs :
                additional keyword arguments

            """

            super().__init__(
                estimator, save_path, {}, None, {},
                train_metrics, val_metrics, None,
                {}, [], save_freq, None, key_mapping,
                logging_type, logging_kwargs, fold, callbacks, start_epoch,
                metric_keys, convert_batch_to_npy_fn, val_freq)

            self._setup(estimator,
                        key_mapping, convert_batch_to_npy_fn)

            for key, val in kwargs.items():
                setattr(self, key, val)

        def _setup(self, estimator, key_mapping, convert_batch_to_npy_fn):
            """
            Defines the Trainers Setup

            Parameters
            ----------
            estimator : :class:`SkLearnEstimator`
                the network to train
            key_mapping : dict
                a dictionary containing the mapping from the ``data_dict`` to
                the actual model's inputs.
                E.g. if a model accepts one input named 'x' and the data_dict
                contains one entry named 'data' this argument would have to
                be ``{'x': 'data'}``
            convert_batch_to_npy_fn : type
                function converting a batch-tensor to numpy

            """

            self.optimizers = create_optims_default()

            super()._setup(estimator, None, {},
                           [], key_mapping, convert_batch_to_npy_fn,
                           estimator.prepare_batch)

            # Load latest epoch file if available
            if os.path.isdir(self.save_path):
                # check all files in directory starting with "checkpoint" and
                # not ending with "_best.pth"
                latest_state_path, latest_epoch = self._search_for_prev_state(
                    self.save_path, [".pkl"])

                # if list is not empty: load previous state
                if latest_state_path is not None:
                    self.update_state(latest_state_path)

                    self.start_epoch = latest_epoch

            self.use_gpu = False
            self.input_device = "cpu"
            self.output_device = "cpu"

            self._prepare_batch = partial(
                self._prepare_batch, input_device=self.input_device,
                output_device=self.output_device)

        def _at_training_begin(self, *args, **kwargs):
            """
            Defines behaviour at beginning of training

            Parameters
            ----------
            *args :
                positional arguments
            **kwargs :
                keyword arguments

            """
            self.save_state(os.path.join(
                self.save_path, "checkpoint_epoch_0"), 0)

        def _get_classes_if_necessary(self, dmgr: BaseDataManager, verbose,
                                      label_key=None):
            """
            Checks if available classes have to be collected before starting
            the training to dynamically build the estimator (not all batches
            contain all classes) and collects them if necessary

            Parameters
            ----------
            dmgr : :class:`BaseDataManager`
                the datamanager to collect the classes from
            verbose : bool
                verbosity
            label_key : str or None
                the key corresponding to the target value inside the data dict

            """

            if label_key is None or not hasattr(self.module, "classes"):
                return
            dset = dmgr.dataset

            if verbose:
                iterable = tqdm(enumerate(dset), unit=' sample', total=len(
                    dset), desc="Creating unique targets to estimate " "classes")

            else:
                iterable = enumerate(dset)

            unique_targets = []

            # iterate over dataset
            for sample_idx, sample in iterable:
                item = sample[label_key]
                if item not in unique_targets:

                    # convert item if necessary
                    if np.isscalar(item):
                        item = np.array([item])
                    unique_targets.append(item)

            # sort and concatenate items and feed variable inside the module
            unique_targets = np.concatenate(list(sorted(unique_targets)))
            self.module.classes = unique_targets

        def train(self, num_epochs, datamgr_train, datamgr_valid=None,
                  val_score_key=None, val_score_mode='highest',
                  reduce_mode='mean', verbose=True, label_key="label"):
            """
            Defines a routine to train a specified number of epochs

            Parameters
            ----------
            num_epochs : int
                number of epochs to train
            datamgr_train : DataManager
                the datamanager holding the train data
            datamgr_valid : DataManager
                the datamanager holding the validation data (default: None)
            val_score_key : str
                the key specifying which metric to use for validation
                (default: None)
            val_score_mode : str
                key specifying what kind of validation score is best
            reduce_mode : str
                'mean','sum','first_only'
            verbose : bool
                whether to show progress bars or not
            label_key : str or None
                key specifiying the value inside the batch dict to use for
                class collection if necessary

            Raises
            ------
            NotImplementedError
                If not overwritten by subclass

            """
            if self.module.iterative_training:

                # estimate classes from validation data
                if datamgr_valid is not None:
                    self._get_classes_if_necessary(datamgr_valid, verbose,
                                                   label_key)
                else:
                    self._get_classes_if_necessary(datamgr_train, verbose,
                                                   label_key)
            else:
                # Setting batchsize to length of dataset and replacing random
                # sampler with replacement by random sampler without replacement
                # ensures, that each sample is present in each batch and only
                # one batch is sampled per epoch
                datamgr_train.batchsize = len(datamgr_train.dataset)
                if isinstance(datamgr_train.sampler, RandomSampler):
                    datamgr_train.sampler = RandomSamplerNoReplacement

                # additionally setting the number of epochs to train ensures,
                # that only one epoch consisting of one batch (which holds the
                # whole dataset) is used for training
                if num_epochs > 1:

                    logging.info(
                        "An epoch number greater than 1 is given, "
                        "but the current module does not support "
                        "iterative training. Falling back to usual "
                        "dataset fitting. For huge datasets, this "
                        "might easily result in out of memory errors!")

                    num_epochs = 1

            return super().train(num_epochs, datamgr_train, datamgr_valid,
                                 val_score_key, val_score_mode, reduce_mode,
                                 verbose)

        def _at_training_end(self):
            """
            Defines Behaviour at end of training: Loads best model if
            available

            Returns
            -------
            :class:`SkLearnEstimator`
                best network

            """
            if os.path.isfile(os.path.join(self.save_path,
                                           'checkpoint_best.pkl')):

                # load best model and return it
                self.update_state(os.path.join(self.save_path,
                                               'checkpoint_best.pkl'))

            return self.module

        def _at_epoch_end(self, metrics_val, val_score_key, epoch, is_best,
                          **kwargs):
            """
            Defines behaviour at beginning of each epoch: Executes all callbacks's
            `at_epoch_end` method and saves current state if necessary

            Parameters
            ----------
            metrics_val : dict
                validation metrics
            val_score_key : str
                validation score key
            epoch : int
                current epoch
            num_epochs : int
                total number of epochs
            is_best : bool
                whether current model is best one so far
            **kwargs :
                keyword arguments

            """

            for cb in self._callbacks:
                self._update_state(cb.at_epoch_end(self,
                                                   val_metrics=metrics_val,
                                                   val_score_key=val_score_key,
                                                   curr_epoch=epoch))

            if epoch % self.save_freq == 0:
                self.save_state(os.path.join(self.save_path,
                                             "checkpoint_epoch_%d.pkl"
                                             % epoch),
                                epoch)

            if is_best:
                self.save_state(os.path.join(self.save_path,
                                             "checkpoint_best.pkl"),
                                epoch)

        def save_state(self, file_name, epoch, **kwargs):
            """
            saves the current state via
            :func:`delira.io.sklearn.save_checkpoint`

            Parameters
            ----------
            file_name : str
                filename to save the state to
            epoch : int
                current epoch (will be saved for mapping back)
            *args :
                positional arguments
            **kwargs :
                keyword arguments

            """
            if not file_name.endswith(".pkl"):
                file_name = file_name + ".pkl"
            save_checkpoint(file_name, self.module, epoch, **kwargs)

        @staticmethod
        def load_state(file_name, *args, **kwargs):
            """
            Loads the new state from file via
            :func:`delira.io.sklearn.load_checkpoint`

            Parameters
            ----------
            file_name : str
                the file to load the state from
            **kwargs : keyword arguments

            Returns
            -------
            dict
                new state

            """

            if not file_name.endswith(".pkl"):
                file_name = file_name + ".pkl"

            return load_checkpoint(file_name, **kwargs)

        def update_state(self, file_name, *args, **kwargs):
            """
            Update internal state from a loaded state

            Parameters
            ----------
            file_name : str
                file containing the new state to load
            *args :
                positional arguments
            **kwargs :
                keyword arguments

            Returns
            -------
            :class:`SkLearnEstimatorTrainer`
                the trainer with a modified state

            """
            self._update_state(self.load_state(file_name, *args, **kwargs))

        def _update_state(self, new_state):
            """
            Update the state from a given new state

            Parameters
            ----------
            new_state : dict
                new state to update internal state from

            Returns
            -------
            :class:`SkLearnEstimatorTrainer`
                the trainer with a modified state

            """

            if "model" in new_state:
                self.module = new_state.pop("model")

            if "epoch" in new_state:
                self.start_epoch = new_state.pop("epoch")

            return super()._update_state(new_state)
