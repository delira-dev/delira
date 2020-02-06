import logging
import os
import pickle
import typing
import warnings

from delira.utils.config import LookupConfig

import numpy as np
from tqdm import tqdm

from .callbacks import AbstractCallback, DefaultLoggingCallback
from .predictor import Predictor
from ..data_loading import Augmenter, DataManager
from ..models import AbstractNetwork
from ..logging import register_logger, make_logger

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
                 metrics: dict,
                 lr_scheduler_cls: type,
                 lr_scheduler_params: dict,
                 gpu_ids: typing.List[int],
                 save_freq: int,
                 optim_fn,
                 key_mapping: dict,
                 logging_type: str,
                 logging_kwargs: dict,
                 logging_callback_cls=DefaultLoggingCallback,
                 logging_frequencies=None,
                 logging_reduce_types=None,
                 fold: int = 0,
                 callbacks: typing.List[AbstractCallback] = None,
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
        metrics : dict, optional
            metrics, which will be evaluated during train and validation phase
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
            If callable: it must be a logging handler backend class
        logging_kwargs : dict
            dictionary containing all logging keyword arguments
        logging_callback_cls : class
            the callback class to create and register for logging
        logging_frequencies : int or dict
                specifies how often to log for each key.
                If int: integer will be applied to all valid keys
                if dict: should contain a frequency per valid key. Missing keys
                will be filled with a frequency of 1 (log every time)
                None is equal to empty dict here.
        logging_reduce_types : str of FunctionType or dict
            if str:
                specifies the reduction type to use. Valid types are
                'last' | 'first' | 'mean' | 'median' | 'max' | 'min'.
                The given type will be mapped to all valid keys.
            if FunctionType:
                specifies the actual reduction function. Will be applied
                for all keys.
            if dict: should contain pairs of valid logging keys and either
                str or FunctionType. Specifies the logging value per key.
                Missing keys will be filles with a default value of 'last'.
                Valid types for strings are
                'last' | 'first' | 'mean' | 'median' | 'max' | 'min'.
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
        if callbacks is None:
            callbacks = []

        # check argument types
        for instance, cls_type in zip([
            network, save_path, losses, optimizer_params, metrics,
            lr_scheduler_params, gpu_ids], [AbstractNetwork, str, dict, dict,
                                            dict, dict, list]):
            if not isinstance(instance, cls_type):
                raise TypeError("%s should be of type %s, but is of type %s"
                                % (instance.__name__, cls_type.__name__,
                                   type(instance).__name__))

        if os.path.isdir(save_path):
            logger.warning(
                "Save Path already exists. Saved Models may be overwritten")
        else:
            os.makedirs(save_path)

        self._fold = fold
        self.start_epoch = start_epoch
        self.save_path = save_path
        self.losses = losses
        self.metrics = metrics
        self.stop_training = False
        self.save_freq = save_freq
        self.metric_keys = metric_keys

        self._tqdm_desc = "Validate"
        self.val_freq = val_freq
        self._global_iter_num = 1
        self._logging_setup_kwargs = {
            "logging_type": logging_type,
            "logging_kwargs": logging_kwargs,
            "logging_callback_cls": logging_callback_cls,
            "logging_frequencies": logging_frequencies,
            "reduce_types": logging_reduce_types}

    def _setup(self, network, lr_scheduler_cls, lr_scheduler_params, gpu_ids,
               key_mapping, convert_batch_to_npy_fn, prepare_batch_fn,
               callbacks):

        super()._setup(network, key_mapping, convert_batch_to_npy_fn,
                       prepare_batch_fn, callbacks)

        self._reinitialize_logging(**self._logging_setup_kwargs)

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

        """
        for cbck in self._callbacks:
            self._update_state(cbck.at_training_begin(self, *args, **kwargs))

        self.save_state(os.path.join(self.save_path, "checkpoint_epoch_%d"
                                     % self.start_epoch))

    def _at_training_end(self, *args, **kwargs):
        """
        Defines the behaviour at the end of the training

        Parameters
        ----------
        *args :
            positional arguments
        **kwargs :
            keyword arguments

        Returns
        -------
        :class:`AbstractNetwork`
            the network with the loaded state

        """
        for cbck in self._callbacks:
            self._update_state(cbck.at_training_end(self, *args, **kwargs))

        return self.module

    def _at_epoch_begin(self, val_score_key, epoch, num_epochs,
                        **kwargs):
        """
        Defines behaviour at beginning of each epoch: Executes all callbacks's
        `at_epoch_begin` method

        Parameters
        ----------
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
            self._update_state(cb.at_epoch_begin(self, val_metrics={},
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

    def _at_iter_begin(self, iter_num, epoch=0, **kwargs):
        """
        Defines the behavior executed at an iteration's begin

        Parameters
        ----------
        iter_num : int
            number of current iter
        epoch : int
            number of current epoch
        **kwargs :
            additional keyword arguments (forwarded to callback calls)

        """
        for cb in self._callbacks:
            self._update_state(cb.at_iter_begin(
                self, iter_num=iter_num,
                curr_epoch=epoch,
                global_iter_num=self._global_iter_num,
                train=True,
                **kwargs,
            ))

    def _at_iter_end(self, iter_num, data_dict, metrics, epoch=0, **kwargs):
        """
        Defines the behavior executed at an iteration's end

        Parameters
        ----------
        iter_num : int
            number of current iter
        data_dict : dict
            dictionary holding input data and predictions
        metrics: dict
            calculated metrics
        epoch : int
            number of current epoch
        **kwargs :
            additional keyword arguments (forwarded to callback calls)

        """

        for cb in self._callbacks:
            self._update_state(cb.at_iter_end(
                self, iter_num=iter_num,
                data_dict=data_dict,
                metrics=metrics,
                curr_epoch=epoch,
                global_iter_num=self._global_iter_num,
                train=True,
                **kwargs,
            ))

        self._global_iter_num += 1

    def _train_single_epoch(self, dmgr_train: DataManager, epoch,
                            verbose=False):
        """
        Trains the network a single epoch

        Parameters
        ----------
        dmgr_train : :class:`DataManager`
            Datamanager to create the data generator
        epoch : int
            current epoch

        """

        metrics, losses = [], []

        batchgen = dmgr_train.get_batchgen(seed=epoch)

        n_batches = dmgr_train.n_batches
        if verbose:
            iterable = tqdm(
                enumerate(batchgen),
                unit=' batch',
                total=n_batches,
                desc='Epoch %d' %
                     epoch)
        else:
            iterable = enumerate(batchgen)

        for iter_num, batch in iterable:
            self._at_iter_begin(epoch=epoch, iter_num=iter_num)

            data_dict = self._prepare_batch(batch)

            _losses, _preds = self.closure_fn(self.module, data_dict,
                                              optimizers=self.optimizers,
                                              losses=self.losses,
                                              fold=self.fold,
                                              iter_num=iter_num)

            data_dict = self._convert_to_npy_fn(**data_dict)[1]
            _preds = self._convert_to_npy_fn(**_preds)[1]

            _metrics = self.calc_metrics(
                LookupConfig(**data_dict, **_preds),
                self.metrics,
                self.metric_keys)

            metrics.append(_metrics)
            losses.append(_losses)

            self._at_iter_end(epoch=epoch, iter_num=iter_num,
                              data_dict={**batch, **_preds},
                              metrics={**_metrics, **_losses},
                              )

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

        for epoch in range(self.start_epoch, num_epochs + 1):

            self._at_epoch_begin(val_score_key, epoch,
                                 num_epochs)

            # train single network epoch
            train_metrics, train_losses = self._train_single_epoch(
                datamgr_train, epoch, verbose=verbose)

            total_metrics = {
                **train_metrics,
                **train_losses}

            # validate network
            if datamgr_valid is not None and (epoch % self.val_freq == 0):
                # next must be called here because self.predict_data_mgr
                # returns a generator (of size 1) and we want to get the
                # first (and only) item
                val_metrics = next(
                    self.predict_data_mgr_cache_metrics_only(
                        datamgr_valid, datamgr_valid.batch_size,
                        metrics=self.metrics,
                        metric_keys=self.metric_keys,
                        verbose=verbose))

                val_metrics = {"val_" + k: v
                               for k, v in val_metrics.items()}

                total_metrics.update(val_metrics)
            _, total_metrics = self._convert_to_npy_fn(**total_metrics)

            for k, v in total_metrics.items():
                total_metrics[k] = reduce_fn(v)

            # check if metric became better
            if val_score_key is not None:
                if val_score_key not in total_metrics:
                    if "val_" + val_score_key not in total_metrics:
                        warnings.warn("val_score_key '%s' not a valid key "
                                      "for validation metrics" %
                                      str(val_score_key), UserWarning)

                        new_val_score = best_val_score

                    else:
                        new_val_score = \
                            total_metrics["val_" + val_score_key]
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

            self._at_epoch_end(total_metrics, val_score_key, epoch,
                               is_best)

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
                        "'at_training_begin' and 'at_training_end'"

        instance_check = isinstance(callback, AbstractCallback)
        attr_check_begin_train = hasattr(callback, "at_training_begin")
        attr_check_end_train = hasattr(callback, "at_training_end")
        attr_check_both_train = attr_check_begin_train and attr_check_end_train

        assert instance_check or attr_check_both_train, assertion_str

        super().register_callback(callback)

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

    @property
    def name(self):
        return os.path.basename(os.path.dirname(os.path.dirname(
            os.path.dirname(self.save_path))))

    def _reinitialize_logging(self, logging_type, logging_kwargs: dict,
                              logging_callback_cls, logging_frequencies,
                              reduce_types):
        """

        Parameters
        ----------
        logging_type : str or callable
            the type of logging. If string: it must be one of
            ["visdom", "tensorboardx"]
            If callable: it must be a logging handler backend class
        logging_kwargs : dict
            dictionary containing all logging keyword arguments
        logging_callback_cls : class
            the callback class to create and register for logging
        logging_frequencies : int or dict
                specifies how often to log for each key.
                If int: integer will be applied to all valid keys
                if dict: should contain a frequency per valid key. Missing keys
                will be filled with a frequency of 1 (log every time)
                None is equal to empty dict here.
        reduce_types : str of FunctionType or dict
            Values are logged in each iteration. This argument specifies,
            how to reduce them to a single value if a logging_frequency
            besides 1 is passed

            if str:
                specifies the reduction type to use. Valid types are
                'last' | 'first' | 'mean' | 'max' | 'min'.
                The given type will be mapped to all valid keys.
            if FunctionType:
                specifies the actual reduction function. Will be applied
                for all keys.
            if dict: should contain pairs of valid logging keys and either
                str or FunctionType. Specifies the logging value per key.
                Missing keys will be filles with a default value of 'last'.
                Valid types for strings are
                'last' | 'first' | 'mean' | 'max' | 'min'.

        """

        from delira.logging import TensorboardBackend, VisdomBackend, \
            BaseBackend

        if isinstance(logging_type, str):
            if logging_type.lower() == "visdom":
                backend_cls = VisdomBackend

            elif logging_type.lower() == "tensorboardx":
                backend_cls = TensorboardBackend

            else:
                raise ValueError("Invalid Logging Type")

        elif issubclass(logging_type, BaseBackend):
            backend_cls = logging_type

        else:
            raise ValueError("Invalid logging_type passed")

        _logging_kwargs = {}

        if backend_cls == VisdomBackend:
            _logging_kwargs.update({"exp_name": "main",
                                    "level": 0})
        elif backend_cls == TensorboardBackend:
            _logging_kwargs.update(
                {
                    "logdir":
                        os.path.join(os.path.dirname(
                            os.path.dirname(self.save_path)),
                            "logs", "run_%02d" % self.fold),
                    "level": 0})

        _logging_kwargs.update(logging_kwargs)

        if "exp_name" in _logging_kwargs.keys():
            _logging_kwargs["exp_name"] = _logging_kwargs["exp_name"] + \
                "_%02d" % self.fold

        # remove prior Trixihandlers and reinitialize it with given logging
        # type
        # This facilitates visualization of multiple splits/fold inside one
        # tensorboard-instance by means of
        # different tf.Summary.FileWriters()

        level = _logging_kwargs.pop("level")

        logger = backend_cls(_logging_kwargs)

        self.register_callback(
            logging_callback_cls(
                logger, level=level,
                logging_frequencies=logging_frequencies,
                reduce_types=reduce_types))

        register_logger(self._callbacks[-1]._logger, self.name)

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
                int(x.rsplit("_", 1)[-1].split(".", 1)[0])
                for x in files])

            latest_state_filename = [x for x in files
                                     if x.startswith("checkpoint_epoch_%d"
                                                     % latest_epoch)][0]
            latest_state_path = os.path.join(path, latest_state_filename)
            return latest_state_path, latest_epoch

        return None, 1

    def register_callback(self, callback: AbstractCallback):
        """
        Registers the passed callback to the trainer,
        after checking it is really a valid callback

        Parameters
        ----------
        callback : AbstractCallback
            the potential callback to register

        Raises
        ------
        AssertionError
            :param:`callback` is not an instance of :class:`AbstractCallback`
            and does not provide the methods `at_iter_begin`, `at_iter_end`,
            `at_epoch_begin` and `at_epoch_end`

        """
        has_all_attrs = True
        for attr in ("epoch",):
            has_all_attrs = has_all_attrs and hasattr(callback,
                                                      "at_%s_begin" % attr)
            has_all_attrs = has_all_attrs and hasattr(callback,
                                                      "at_%s_end" % attr)

        assert has_all_attrs, "Given callback is not valid; Must be " \
                              "instance of AbstractCallback or provide " \
                              "functions 'at_epoch_begin' and 'at_epoch_end'"
        super().register_callback(callback)
