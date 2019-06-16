import logging
import os
import pickle
import typing

import numpy as np
from tqdm import tqdm

from delira.logging import TrixiHandler
from .callbacks import AbstractCallback
from .predictor import Predictor
from ..data_loading.data_manager import Augmenter
from ..models import AbstractNetwork

logger = logging.getLogger(__name__)


class BaseNetworkTrainer(Predictor):
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
                 lr_scheduler_cls: type,
                 lr_scheduler_params: dict,
                 gpu_ids: typing.List[int],
                 save_freq: int,
                 optim_fn,
                 key_mapping: dict,
                 logging_type: str,
                 logging_kwargs: dict,
                 fold: int,
                 callbacks: typing.List[AbstractCallback],
                 start_epoch=1,
                 metric_keys=None,
                 convert_batch_to_npy_fn=lambda x: x,
                 val_freq=1,
                 **kwargs
                 ):
        """

        Parameters
        ----------
        network : :class:`AbstractTfNetwork`
            the network to train
        save_path : str
            path to save networks to
        losses : dict
            dictionary containing the training losses
        optimizer_cls : subclass of tf.train.Optimizer
            optimizer class implementing the optimization algorithm of choice
        optimizer_params : dict
            keyword arguments passed to optimizer during construction
        train_metrics : dict, optional
            metrics, which will be evaluated during train phase
            (should work on numpy arrays)
        val_metrics : dict, optional
            metrics, which will be evaluated during test phase
            (should work on numpy arrays)
        lr_scheduler_cls : Any
            learning rate schedule class: must implement step() method
        lr_scheduler_params : dict
            keyword arguments passed to lr scheduler during construction
        gpu_ids : list
            list containing ids of GPUs to use; if empty: use cpu instead
        save_freq : int
            integer specifying how often to save the current model's state.
            State is saved every state_freq epochs
        optim_fn : function
            creates a dictionary containing all necessary optimizers
        key_mapping : dict
            a dictionary containing the mapping from the ``data_dict`` to
            the actual model's inputs.
            E.g. if a model accepts one input named 'x' and the data_dict
            contains one entry named 'data' this argument would have to
            be ``{'x': 'data'}``
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
            the batch_dict keys to use for each metric to calculate.
            Should contain a value for each key in ``metrics``.
            If no values are given for a key, per default ``pred`` and
            ``label`` will be used for metric calculation
        convert_batch_to_npy_fn : type, optional
            function converting a batch-tensor to numpy, per default this is
            the identity function
        val_freq : int
            validation frequency specifying how often to validate the trained
            model (a value of 1 denotes validating every epoch,
            a value of 2 denotes validating every second epoch etc.);
            defaults to 1
        **kwargs :
            Additional keyword arguments

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
        assert isinstance(lr_scheduler_params, dict)
        assert isinstance(gpu_ids, list)

        if os.path.isdir(save_path):
            logger.warning(
                "Save Path already exists. Saved Models may be overwritten")
        else:
            os.makedirs(save_path)

        self._callbacks = []
        self._fold = fold
        self.start_epoch = start_epoch
        self.save_path = save_path
        self.losses = losses
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.stop_training = False
        self.save_freq = save_freq
        self.metric_keys = metric_keys

        for cbck in callbacks:
            self.register_callback(cbck)

        self._reinitialize_logging(logging_type, logging_kwargs)
        self._tqdm_desc = "Validate"
        self.val_freq = val_freq

    def _setup(self, network, lr_scheduler_cls, lr_scheduler_params, gpu_ids,
               key_mapping, convert_batch_to_npy_fn, prepare_batch_fn):

        super()._setup(network, key_mapping, convert_batch_to_npy_fn,
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

    def _train_single_epoch(self, batchgen: Augmenter, epoch,
                            verbose=False):
        """
        Trains the network a single epoch

        Parameters
        ----------
        batchgen : :class:`Augmenter`
            Generator yielding the training batches
        epoch : int
            current epoch

        """

        metrics, losses = [], []

        n_batches = batchgen.num_batches
        if verbose:
            iterable = tqdm(
                enumerate(batchgen),
                unit=' batch',
                total=n_batches,
                desc='Epoch %d' %
                     epoch)
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

        total_losses, total_metrics = {}, {}

        for _metrics in metrics:
            for key, val in _metrics.items():
                if key in total_metrics:
                    total_metrics[key].append(val)
                else:
                    total_metrics[key] = [val]

        for _losses in losses:
            for key, val in _losses.items():
                if key in total_losses:
                    total_losses[key].append(val)
                else:
                    total_losses[key] = [val]

        return total_metrics, total_losses

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
        verbose : bool
            whether to show progress bars or not

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
            def reduce_fn(batch):
                return np.mean(batch)
        elif reduce_mode == 'sum':
            def reduce_fn(batch):
                return np.sum(batch)
        elif reduce_mode == 'first_only':
            def reduce_fn(batch):
                return batch[0]
        elif reduce_mode == 'last_only':
            def reduce_fn(batch):
                return batch[-1]
        else:
            raise ValueError("No valid reduce mode given")

        metrics_val = {}

        val_metric_fns = {}

        for k, v in self.val_metrics.items():
            if not k.startswith("val_"):
                k = "val_" + k

            val_metric_fns[k] = v

        if self.metric_keys is None:
            val_metric_keys = None

        else:
            val_metric_keys = {}
            for k, v in self.metric_keys.items():
                if not k.startswith("val_"):
                    k = "val_" + k

                val_metric_keys[k] = v

        for epoch in range(self.start_epoch, num_epochs + 1):

            self._at_epoch_begin(metrics_val, val_score_key, epoch,
                                 num_epochs)

            batch_gen_train = datamgr_train.get_batchgen(seed=epoch)

            # train single network epoch
            train_metrics, train_losses = self._train_single_epoch(
                batch_gen_train, epoch, verbose=verbose)

            total_metrics = {
                **train_metrics,
                **train_losses}

            # validate network
            if datamgr_valid is not None and (epoch % self.val_freq == 0):
                # next must be called here because self.predict_data_mgr
                # returns a generator (of size 1) and we want to get the first
                # (and only) item
                val_metrics = next(
                    self.predict_data_mgr_cache_metrics_only(
                        datamgr_valid, datamgr_valid.batch_size,
                        metrics=val_metric_fns, metric_keys=val_metric_keys,
                        verbose=verbose))

                total_metrics.update(val_metrics)

            for k, v in total_metrics.items():
                total_metrics[k] = reduce_fn(v)

            # check if metric became better
            if val_score_key is not None:
                if val_score_key not in total_metrics:
                    if "val_" + val_score_key not in total_metrics:
                        logger.warning(
                            "val_score_key '%s' not a valid key for \
                                    validation metrics" % str(val_score_key))

                        new_val_score = best_val_score

                    else:
                        new_val_score = total_metrics["val_" + val_score_key]
                        val_score_key = "val_" + val_score_key
                else:
                    new_val_score = total_metrics.get(val_score_key)

            if new_val_score != best_val_score:
                is_best = self._is_better_val_scores(
                    best_val_score, new_val_score, val_score_mode)

                # set best_val_score to new_val_score if is_best
                if is_best:
                    best_val_score = new_val_score

                if is_best and verbose:
                    logging.info("New Best Value at Epoch %03d : %03.3f" %
                                 (epoch, best_val_score))

            # log metrics and loss values
            for key, val in total_metrics.items():
                logging.info({"value": {"value": val, "name": key
                                        }})

            self._at_epoch_end(total_metrics, val_score_key, epoch, is_best)

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
        instance_check = isinstance(callback, AbstractCallback)
        attr_check_begin = hasattr(callback, "at_epoch_begin")
        attr_check_end = hasattr(callback, "at_epoch_end")
        attr_check_both = attr_check_begin and attr_check_end

        assert instance_check or attr_check_both, assertion_str

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
        :class:`BaseNetworkTrainer`
            the trainer with a modified state

        """
        for key, val in new_state.items():
            if key.startswith("__") and key.endswith("__"):
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
        :class:`BaseNetworkTrainer`
            the trainer with a modified state

        """
        self._update_state(self.load_state(file_name, *args, **kwargs))

    @staticmethod
    def _is_better_val_scores(old_val_score, new_val_score,
                              mode='highest'):
        """
        Check whether the new val score is better than the old one
        with respect to the optimization goal

        Parameters
        ----------
        old_val_score :
            old validation score
        new_val_score :
            new validation score
        mode: str
            String to specify whether a higher or lower validation score is
            optimal; must be in ['highest', 'lowest']

        Returns
        -------
        bool
            True if new score is better, False otherwise
        """

        assert mode in ['highest', 'lowest'], "Invalid Comparison Mode"

        if mode == 'highest':
            return new_val_score > old_val_score
        elif mode == 'lowest':
            return new_val_score < old_val_score

    def _reinitialize_logging(self, logging_type, logging_kwargs: dict):
        from ..logging import TensorboardXLoggingHandler, VisdomLoggingHandler

        if isinstance(logging_type, str):
            if logging_type.lower() == "visdom":
                logging_cls = VisdomLoggingHandler

            elif logging_type.lower() == "tensorboardx":
                logging_cls = TensorboardXLoggingHandler

            else:
                raise ValueError("Invalid Logging Type")

        else:
            logging_cls = logging_type

        if logging_cls == VisdomLoggingHandler:
            _logging_kwargs = {"exp_name": "main",
                               "level": 0}
        elif logging_cls == TensorboardXLoggingHandler:
            _logging_kwargs = {"log_dir": self.save_path,
                               "level": 0}

        _logging_kwargs.update(logging_kwargs)

        if "exp_name" in _logging_kwargs.keys():
            _logging_kwargs["exp_name"] = _logging_kwargs["exp_name"] + \
                "_%02d" % self.fold

        # remove prior Trixihandlers and reinitialize it with given logging
        # type
        # This facilitates visualization of multiple splits/fold inside one
        # tensorboard-instance by means of
        # different tf.Summary.FileWriters()

        root_logger = logging.getLogger()
        new_handlers = []
        for handler in root_logger.handlers:
            if isinstance(handler, TrixiHandler):
                handler.close()
            else:
                new_handlers.append(handler)

        root_logger.handlers = []

        new_handlers.append(
            logging_cls(**_logging_kwargs)
        )
        logging.basicConfig(level=logging.INFO,
                            handlers=new_handlers)

    @staticmethod
    def _search_for_prev_state(path, extensions=None):
        """
        Helper function to search in a given path for previous epoch states
        (indicated by extensions)

        Parameters
        ----------
        path : str
            the path to search in
        extensions : list
            list of strings containing valid file extensions for checkpoint
            files

        Returns
        -------
        str
            the file containing the latest checkpoint (if available)
        None
            if no latst checkpoint was found
        int
            the latest epoch (1 if no checkpoint was found)

        """
        if extensions is None:
            extensions = []
        files = []
        for file in os.listdir(path):
            for ext in extensions:
                if not ext.startswith("."):
                    ext = "." + ext

                if not file.endswith(ext):
                    continue

                if not file.startswith("checkpoint"):
                    continue

                if file.endswith("_best" + ext):
                    continue

                files.append(file)
                break

        if files:
            latest_epoch = max([
                int(x.rsplit("_", 1)[-1].rsplit(".", 1)[0])
                for x in files])

            latest_state_path = [x for x in files
                                 if x.startswith("checkpoint_%d"
                                                 % latest_epoch)][0]

            return latest_state_path, latest_epoch

        return None, 1
