from abc import abstractmethod
import logging
import pickle
import typing

from batchgenerators.dataloading import MultiThreadedAugmenter

from .predictor import Predictor
from .callbacks import AbstractCallback
from ..models import AbstractNetwork

import numpy as np
import os
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AbstractNetworkTrainer(Predictor):
    """
    Defines a Base API and basic functions for Network Trainers

    See Also
    --------
    :class:`PyTorchNetworkTrainer`
    :class:`TfNetworkTrainer`

    """

    __KEYS_TO_GUARD = ["use_gpu",
                       "input_device",
                       "output_device",
                       "_callbacks"]

    def __init__(self,
                 network: AbstractNetwork,
                 save_path: str,
                 losses: dict,
                 optimizer_cls: type,
                 optimizer_params: dict,
                 train_metrics: dict,
                 val_metrics: dict,
                 val_dataset_metrics: dict,
                 lr_scheduler_cls: type,
                 lr_scheduler_params: dict,
                 gpu_ids: typing.List[int],
                 save_freq: int,
                 optim_fn,
                 fold: int,
                 callbacks: typing.List[AbstractCallback],
                 start_epoch: int,
                 convert_batch_to_npy_fn=lambda x: x,
                 **kwargs
                 ):
        """

        Parameters
        ----------
        fold : int
            current fold in (default: 0)
        callbacks : list
            list of callbacks to register

        """

        # explicity not call self._setup here to reuse the __init__ of
        # abstract class. self._setup has to be called in subclass

        # check argument types
        assert isinstance(network, AbstractNetwork)
        assert isinstance(save_path, str)
        assert isinstance(losses, dict)
        assert isinstance(optimizer_params, dict)
        assert isinstance(train_metrics, dict)
        assert isinstance(val_metrics, dict)
        assert isinstance(val_dataset_metrics, dict)
        assert isinstance(lr_scheduler_params, dict)
        assert isinstance(gpu_ids, list)

        if os.path.isdir(save_path):
            logger.warning(
                "Save Path already exists. Saved Models may be overwritten")
        else:
            os.makedirs(save_path)

        self._callbacks = []
        self._fold = fold
        self.start_epoch = 0
        self.save_path = save_path
        self.losses = losses
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.val_dataset_metrics = val_dataset_metrics
        self.stop_training = False
        self.save_freq = save_freq

        for cbck in callbacks:
            self.register_callback(cbck)

    def _setup(self, network, lr_scheduler_cls, lr_scheduler_params, gpu_ids,
               convert_batch_to_npy_fn, prepare_batch_fn):

        super()._setup(network, convert_batch_to_npy_fn,
                       prepare_batch_fn)

        self.closure_fn = network.closure
        
        # optimizers must exist before calling _setup()
        if lr_scheduler_cls is not None:
            for key, optim in self.optimizers.items():
                if not issubclass(lr_scheduler_cls, AbstractCallback):
                    logger.warning("lr_scheduler_cls is not a callback.")
                self.register_callback(lr_scheduler_cls(optim,
                                                        **lr_scheduler_params))

        if gpu_ids:
            self.use_gpu = True
        else:
            self.use_gpu = False

    def _at_training_begin(self, *args, **kwargs):
        """
        Defines the behaviour at beginnig of the training

        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments

        Raises
        ------
        NotImplementedError
            If not overwritten by subclass

        """
        self.save_state(os.path.join(self.save_path, "checkpoint_epoch_0"))

    def _at_training_end(self, *args, **kwargs):
        """
        Defines the behaviour at the end of the training

        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments

        Raises
        ------
        NotImplementedError
            If not overwritten by subclass

        """
        return self.module

    def _at_epoch_begin(self, metrics_val, val_score_key, epoch, num_epochs,
                        **kwargs):
        """
        Defines behaviour at beginning of each epoch: Executes all callbacks's
        `at_epoch_begin` method

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
        **kwargs :
            keyword arguments

        """

        # execute all callbacks
        for cb in self._callbacks:
            self._update_state(cb.at_epoch_begin(self, val_metrics=metrics_val,
                                                 val_score_key=val_score_key,
                                                 curr_epoch=epoch))

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
        **kwargs :
            keyword arguments

        """

        for cb in self._callbacks:
            self._update_state(cb.at_epoch_end(self, val_metrics=metrics_val,
                                               val_score_key=val_score_key,
                                               curr_epoch=epoch))

        if epoch % self.save_freq == 0:
            self.save_state(os.path.join(self.save_path,
                                         "checkpoint_epoch_%d" % epoch))

        if is_best:
            self.save_state(os.path.join(self.save_path,
                                         "checkpoint_best"))

    def _train_single_epoch(self, batchgen: MultiThreadedAugmenter, epoch,
                            verbose=False):
        """
        Trains the network a single epoch

        Parameters
        ----------
        batchgen : MultiThreadedAugmenter
            Generator yielding the training batches
        epoch : int
            current epoch

        """

        metrics, losses = [], []

        n_batches = batchgen.generator.num_batches * batchgen.num_processes
        if verbose:
            iterable = tqdm(enumerate(batchgen), unit=' batch', total=n_batches,
                            desc='Epoch %d' % epoch)
        else:
            iterable = enumerate(batchgen)

        for batch_nr, batch in iterable:

            data_dict = self._prepare_batch(batch)

            _metrics, _losses, _ = self.closure_fn(self.module, data_dict,
                                                   optimizers=self.optimizers,
                                                   losses=self.losses,
                                                   metrics=self.train_metrics,
                                                   fold=self.fold,
                                                   batch_nr=batch_nr)
        metrics.append(_metrics)
        losses.append(_losses)

        batchgen._finish()

        return {
            k: np.vstack([self._convert_batch_to_npy_fn(_metrics[k])
                          for _metrics in metrics])
            for k in metrics[0].keys()
        }, {
            k: np.vstack([self._convert_batch_to_npy_fn(_losses[k])
                          for _losses in losses])
            for k in losses[0].keys()
        }

    def train(self, num_epochs, datamgr_train, datamgr_valid=None,
              val_score_key=None, val_score_mode='highest', reduce_mode='mean',
              verbose=True):
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

        Raises
        ------
        NotImplementedError
            If not overwritten by subclass

        """
        self._at_training_begin()

        if val_score_mode == 'highest':
            best_val_score = 0
        elif val_score_mode == 'lowest':
            best_val_score = float('inf')
        else:
            best_val_score = None

        is_best = False
        new_val_score = best_val_score

        if reduce_mode == 'mean':
            def reduce_fn(batch): return batch.mean()
        elif reduce_mode == 'sum':
            def reduce_fn(batch): return batch.sum()
        elif reduce_mode == 'first_only':
            def reduce_fn(batch): return batch[0]
        elif reduce_mode == 'last_only':
            def reduce_fn(batch): return batch[-1]
        else:
            raise ValueError("No valid reduce mode given")

        metrics_val = {}

        for epoch in range(self.start_epoch, num_epochs+1):

            self._at_epoch_begin(metrics_val, val_score_key, epoch,
                                 num_epochs)

            batch_gen_train = datamgr_train.get_batchgen(seed=epoch)

            train_metrics, train_losses = self._train_single_epoch(
                batch_gen_train, epoch, verbose=verbose)

            val_predictions, val_inputs = self.predict_data_mgr(
                datamgr_valid, datamgr_valid.batch_size, verbose=verbose)

            metrics_val = self.calc_metrics(
                {k: v for k,
                 v in val_inputs.items() if k != "data"},
                *val_predictions,
                metrics={**self.val_metrics, **self.val_dataset_metrics})

            total_metrics = {
                **{"val_" + k: v for k, v in metrics_val.items()},
                **train_metrics,
                **train_losses}

            if val_score_key is not None:
                if val_score_key not in total_metrics:
                    if "val_" + val_score_key not in total_metrics:
                        logger.warning("val_score_key '%s' not a valid key for \
                                    validation metrics ")

                        new_val_score = best_val_score

                    else:
                        new_val_score = total_metrics["val_" + val_score_key]
                else:
                    new_val_score = total_metrics.get(val_score_key)

            if new_val_score != best_val_score:
                is_best = self._is_better_val_scores(
                    best_val_score, new_val_score, val_score_mode)

                # set best_val_score to new_val_score if is_best
                best_val_score = int(is_best) * new_val_score + \
                    (1 - int(is_best)) * best_val_score

                if is_best and verbose:
                    logging.info("New Best Value at Epoch %03d : %03.3f" %
                                 (epoch, best_val_score))

            for key, val in total_metrics.items():
                logging.info({"value": {"value": reduce_fn(val), "name": key,
                                        "env_appendix": "_%02d" % self.fold
                                        }})

            self._at_epoch_end(metrics_val, val_score_key, epoch, is_best)

            is_best = False

            # stop training (might be caused by early stopping)
            if self.stop_training:
                break

        return self._at_training_end()

    @property
    def fold(self):
        """
        Get current fold

        Returns
        -------
        int
            current fold

        """
        return self._fold

    @fold.setter
    def fold(self, fold):
        """
        Set the current fold

        Parameters
        ----------
        fold : int
            new fold

        Raises
        ------
        ValueError
            if `fold` is not covertable to :obj:`int`

        """
        try:
            self._fold = int(fold)

        except ValueError as e:
            logger.error(e)
            raise e

    def register_callback(self, callback: AbstractCallback):
        """
        Register Callback to Trainer

        Parameters
        ----------
        callback : :class:`AbstractCallback`
            the callback to register

        Raises
        ------
        AssertionError
            `callback` is not an instance of :class:`AbstractCallback` and has
            not both methods ['at_epoch_begin', 'at_epoch_end']

        """
        assertion_str = "Given callback is not valid; Must be instance of " \
            "AbstractCallback or provide functions " \
            "'at_epoch_begin' and 'at_epoch_end'"

        assert isinstance(callback, AbstractCallback) or \
            (hasattr(callback, "at_epoch_begin")
             and hasattr(callback, "at_epoch_end")), assertion_str

        self._callbacks.append(callback)

    def save_state(self, file_name, *args, **kwargs):
        """
        saves the current state

        Parameters
        ----------
        file_name : str
            filename to save the state to
        *args :
            positional arguments
        **kwargs :
            keyword arguments

        """
        with open(file_name, "wb") as f:
            pickle.dump(vars(self), f, *args, **kwargs)

    @staticmethod
    def load_state(file_name, *args, **kwargs):
        """
        Loads the new state from file

        Parameters
        ----------
        file_name : str
            the file to load the state from
        *args :
            positional arguments
        **kwargs : keyword arguments

        Returns
        -------
        dict
            new state

        """
        with open(file_name, "rb") as f:
            new_state = pickle.load(f, *args, **kwargs)

        return new_state

    def _update_state(self, new_state):
        """
        Update the state from a given new state

        Parameters
        ----------
        new_state : dict
            new state to update internal state from

        Returns
        -------
        :class:`AbstractNetworkTrainer`
            the trainer with a modified state

        """
        for key, val in new_state.items():
            if (key.startswith("__") and key.endswith("__")):
                continue

            try:
                setattr(self, key, val)

            except PermissionError:
                logger.error("Trying to overwrite attribute %s of "
                             "NetworkTrainer, which is not allowed!" % key)

        return self

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
        :class:`AbstractNetworkTrainer`
            the trainer with a modified state

        """
        self._update_state(self.load_state(file_name, *args, **kwargs))
