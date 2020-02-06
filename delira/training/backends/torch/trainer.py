import logging
import os
from functools import partial
import warnings

import torch
from batchgenerators.dataloading import MultiThreadedAugmenter

from delira.io.torch import load_checkpoint_torch, save_checkpoint_torch
from delira.models.backends.torch import AbstractPyTorchNetwork, \
    DataParallelPyTorchNetwork

from delira.training.base_trainer import BaseNetworkTrainer

from delira.training.backends.torch.utils import create_optims_default
from delira.training.backends.torch.utils import convert_to_numpy
from delira.training.callbacks.logging_callback import DefaultLoggingCallback


logger = logging.getLogger(__name__)


class PyTorchNetworkTrainer(BaseNetworkTrainer):
    """
    Train and Validate a Network

    See Also
    --------
    :class:`AbstractNetwork`

    """

    def __init__(self,
                 network: AbstractPyTorchNetwork,
                 save_path: str,
                 key_mapping,
                 losses=None,
                 optimizer_cls=None,
                 optimizer_params=None,
                 metrics=None,
                 lr_scheduler_cls=None,
                 lr_scheduler_params=None,
                 gpu_ids=None,
                 save_freq=1,
                 optim_fn=create_optims_default,
                 logging_type="tensorboardx",
                 logging_kwargs=None,
                 logging_callback_cls=DefaultLoggingCallback,
                 logging_frequencies=None,
                 logging_reduce_types=None,
                 fold=0,
                 callbacks=None,
                 start_epoch=1,
                 metric_keys=None,
                 convert_batch_to_npy_fn=convert_to_numpy,
                 mixed_precision=False,

                 mixed_precision_kwargs={"opt_level": "O1",
                                         "cast_model_type": None,
                                         "patch_torch_functions": None,
                                         "master_weights": None,
                                         "loss_scale": None,
                                         "cast_model_outputs": None,
                                         "num_losses": 1,
                                         "verbosity": 1},
                 val_freq=1,
                 ** kwargs):
        """

        Parameters
        ----------
        network : :class:`AbstractPyTorchNetwork`
            the network to train
        save_path : str
            path to save networks to
        key_mapping : dict
            a dictionary containing the mapping from the ``data_dict`` to
            the actual model's inputs.
            E.g. if a model accepts one input named 'x' and the data_dict
            contains one entry named 'data' this argument would have to
            be ``{'x': 'data'}``
        losses : dict
            dictionary containing the training losses
        optimizer_cls : subclass of tf.train.Optimizer
            optimizer class implementing the optimization algorithm of
            choice
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
        logging_type : str or callable
            the type of logging. If string: it must be one of
            ["visdom", "tensorboardx"]
            If callable: it must be a logging handler class
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
            dict specifying which batch_dict entry to use for which metric
            as target; default: None, which will result in key "label"
            for all metrics
        convert_batch_to_npy_fn : type, optional
            function converting a batch-tensor to numpy, per default this
            is a function, which detaches the tensor, moves it to cpu and
            then calls ``.numpy()`` on it
        mixed_precision : bool
            whether to use mixed precision or not (False per default)
        mixed_precision_kwargs : dict
            additional keyword arguments for mixed precision
            from apex.amp.frontend:
                opt_level : str
                    Pure or mixed precision optimization level. Accepted
                    values are "O0", "O1", "O2", and "O3":
                        O0:  Pure FP32 training.
                        O1:  Insert automatic casts around Pytorch
                            functions and Tensor methods.
                        O2:  FP16 training with FP32 batchnorm and FP32
                            master weights
                        O3:  Pure FP16 training.

                cast_model_type : :class:`torch.dtype`
                    Optional property override for model dtype;
                    default: None
                patch_torch_functions : bool
                    Optional property override.
                keep_batchnorm_fp32 : bool or str
                    Optional property override.  If passed as a string,
                    must be the string "True" or "False".
                master_weights : bool
                    Optional property override; whether to create master
                    weights or not
                loss_scale : float or str
                    Optional property override.  If passed as a string,
                    must be a string representing a number, e.g., "128.0",
                    or the string "dynamic".
                cast_model_outputs : :class:`torch.dtype`
                    Option to ensure that the outputs of your model(s)
                    are always cast to a particular type regardless
                    of ``opt_level``.
                num_losses : int
                    Option to tell Amp in advance how many losses/backward
                    passes you plan to use.  When used in conjunction with
                    the ``loss_id`` argument to ``amp.scale_loss``, enables
                    Amp to use a different loss scale per loss/backward
                    pass, which can improve stability. See
                    "Multiple models/optimizers/losses" under
                    "Advanced Amp Usage" for examples.  If ``num_losses``
                    is left to 1, Amp will still support multiple
                    losses/backward passes, but use a single global
                    loss scale for all of them; default: 1
                verbosity : int
                    Set to 0 to suppress Amp-related output; default: 1
        val_freq : int
            validation frequency specifying how often to validate the
            trained model (a value of 1 denotes validating every epoch,
            a value of 2 denotes validating every second epoch etc.);
            defaults to 1
        **kwargs :
            additional keyword arguments

        """

        if callbacks is None:
            callbacks = []
        if logging_kwargs is None:
            logging_kwargs = {}
        if gpu_ids is None:
            gpu_ids = []
        if lr_scheduler_params is None:
            lr_scheduler_params = {}
        if metrics is None:
            metrics = {}
        if optimizer_params is None:
            optimizer_params = {}

        super().__init__(network=network,
                         save_path=save_path,
                         losses=losses,
                         optimizer_cls=optimizer_cls,
                         optimizer_params=optimizer_params,
                         metrics=metrics,
                         lr_scheduler_cls=lr_scheduler_cls,
                         lr_scheduler_params=lr_scheduler_params,
                         gpu_ids=gpu_ids,
                         save_freq=save_freq,
                         optim_fn=optim_fn,
                         key_mapping=key_mapping,
                         logging_type=logging_type,
                         logging_kwargs=logging_kwargs,
                         logging_callback_cls=logging_callback_cls,
                         logging_frequencies=logging_frequencies,
                         logging_reduce_types=logging_reduce_types,
                         fold=fold,
                         callbacks=callbacks,
                         start_epoch=start_epoch,
                         metric_keys=metric_keys,
                         convert_batch_to_npy_fn=convert_batch_to_npy_fn,
                         val_freq=val_freq,
                         **kwargs
                         )

        self._setup(network, optim_fn, optimizer_cls, optimizer_params,
                    lr_scheduler_cls, lr_scheduler_params, gpu_ids,
                    key_mapping, convert_batch_to_npy_fn,
                    mixed_precision, mixed_precision_kwargs, callbacks)

        for key, val in kwargs.items():
            setattr(self, key, val)

    def _setup(self, network, optim_fn, optimizer_cls, optimizer_params,
               lr_scheduler_cls, lr_scheduler_params, gpu_ids,
               key_mapping, convert_batch_to_npy_fn, mixed_precision,
               mixed_precision_kwargs, callbacks):
        """
        Defines the Trainers Setup

        Parameters
        ----------
        network : :class:`AbstractPyTorchNetwork`
            the network to train
        optim_fn : function
            creates a dictionary containing all necessary optimizers
        optimizer_cls : subclass of torch.optim.Optimizer
            optimizer class implementing the optimization algorithm of
            choice
        optimizer_params : dict
        lr_scheduler_cls : Any
            learning rate schedule class: must implement step() method
        lr_scheduler_params : dict
            keyword arguments passed to lr scheduler during construction
        gpu_ids : list
            list containing ids of GPUs to use; if empty: use cpu instead
        convert_batch_to_npy_fn : type
            function converting a batch-tensor to numpy
        mixed_precision : bool
            whether to use mixed precision or not (False per default)
        mixed_precision_kwargs : dict
            additional keyword arguments for mixed precision
        callbacks : list
            initial callbacks to register

        """

        self.optimizers = optim_fn(network, optimizer_cls,
                                   **optimizer_params)

        super()._setup(network, lr_scheduler_cls, lr_scheduler_params,
                       gpu_ids, key_mapping, convert_batch_to_npy_fn,
                       network.prepare_batch, callbacks)

        # Load latest epoch file if available
        if os.path.isdir(self.save_path):
            latest_state_path, latest_epoch = self._search_for_prev_state(
                self.save_path)

            if latest_state_path is not None:

                # if pth file does not exist, load pt file instead
                if not os.path.isfile(latest_state_path):
                    latest_state_path = latest_state_path[:-1]

                logger.info("Attempting to load state from previous \
                            training from %s" % latest_state_path)
                try:
                    self.update_state(latest_state_path)
                except KeyError:
                    logger.warning("Previous State could not be loaded, \
                                although it exists.Training will be \
                                restarted")

                self.start_epoch = latest_epoch

        if gpu_ids and torch.cuda.is_available():
            self.use_gpu = True
            if (len(gpu_ids) > 1) and (torch.cuda.device_count() > 1):
                # use GPU 0 as default input GPU
                self.input_device = torch.device("cuda:%d" % gpu_ids[0])

                # Train on multiple GPUs and use GPU 0 as output device
                self.module = DataParallelPyTorchNetwork(self.module.to(
                    self.input_device),
                    device_ids=gpu_ids,
                    output_device=gpu_ids[1])

                # use GPU 1 as default output GPU for balanced GPU usage
                self.output_device = torch.device("cuda:%d" % gpu_ids[1])
            else:
                # use the only available GPU as input device
                self.input_device = torch.device("cuda:%d" % gpu_ids[0])
                self.module = self.module.to(self.input_device)

                # use GPU 0 as output device as output device
                self.output_device = torch.device("cuda:%d" % gpu_ids[0])
        else:
            self.use_gpu = False
            self.input_device = torch.device("cpu")
            self.output_device = torch.device("cpu")
            self.module = self.module.to(self.input_device)

        self._prepare_batch = partial(
            self._prepare_batch, input_device=self.input_device,
            output_device=self.output_device)

        try:
            # use apex for mixed precision if installed
            from apex import amp

            # extract optimizers and corresponding keys
            # (in case dict is not ordered)
            _optim_keys = list(self.optimizers.keys())
            _optims = list(self.optimizers[k] for k in _optim_keys)

            # wrap model and register optimizers for mixed precision
            self.module, _optims = amp.initialize(self.module, _optims,
                                                  mixed_precision,
                                                  **mixed_precision_kwargs)
            for k, v in zip(_optim_keys, _optims):
                self.optimizers[k] = v

        except (ImportError, RuntimeError) as e:
            warnings.warn(
                "Either APEX can't be imported correctly or a value "
                "missmatch occured. Switching to default FP32 "
                "training insted. The following Exception occured:"
                "\n%s" %
                str(e))

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
                                     % self.start_epoch), self.start_epoch)

    def _at_training_end(self, *args, **kwargs):
        """
        Defines Behaviour at end of training: Loads best model if
        available

        Returns
        -------
        :class:`AbstractPyTorchNetwork`
            best network

        """
        if os.path.isfile(os.path.join(self.save_path,
                                       'checkpoint_best.pt')):

            # load best model and return it
            self.update_state(os.path.join(self.save_path,
                                           'checkpoint_best.pt'))

        return super()._at_training_end(*args, **kwargs)

    def _at_epoch_end(self, metrics_val, val_score_key, epoch, is_best,
                      **kwargs):
        """
        Defines behaviour at beginning of each epoch:
        Executes all callbacks's `at_epoch_end` method and saves current
        state if necessary

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
            self._update_state(
                cb.at_epoch_end(
                    self,
                    val_metrics=metrics_val,
                    val_score_key=val_score_key,
                    curr_epoch=epoch))

        if epoch % self.save_freq == 0:
            self.save_state(os.path.join(self.save_path,
                                         "checkpoint_epoch_%d.pt" % epoch),
                            epoch)

        if is_best:
            self.save_state(os.path.join(self.save_path,
                                         "checkpoint_best.pt"),
                            epoch)

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

        self.module.train()

        return super()._train_single_epoch(batchgen, epoch,
                                           verbose=verbose)

    def predict_data_mgr(self, datamgr, batchsize=None, metrics=None,
                         metric_keys=None, verbose=False, **kwargs):
        """
        Defines a routine to predict data obtained from a batchgenerator

        Parameters
        ----------
        datamgr : :class:`DataManager`
            Manager producing a generator holding the batches
        batchsize : int
            Artificial batchsize (sampling will be done with batchsize
            1 and sampled data will be stacked to match the artificial
            batchsize)(default: None)
        metrics : dict
            the metrics to calculate
        metric_keys : dict
            the ``batch_dict`` items to use for metric calculation
        verbose : bool
            whether to show a progress-bar or not, default: False
        **kwargs :
            additional keyword arguments

        Returns
        -------
        dict
            predictions
        dict
            calculated metrics

        """
        self.module.eval()

        if metrics is None:
            metrics = {}

        return super().predict_data_mgr(datamgr, batchsize, metrics,
                                        metric_keys, verbose, **kwargs)

    def save_state(self, file_name, epoch, **kwargs):
        """
        saves the current state via :func:`delira.io.torch.save_checkpoint`

        Parameters
        ----------
        file_name : str
            filename to save the state to
        epoch : int
            current epoch (will be saved for mapping back)
        **kwargs :
            keyword arguments

        """
        if not (file_name.endswith(".pth") or file_name.endswith(".pt")):
            file_name = file_name + ".pt"
        save_checkpoint_torch(file_name, self.module, self.optimizers, epoch,
                              **kwargs)

    @staticmethod
    def load_state(file_name, **kwargs):
        """
        Loads the new state from file via
        :func:`delira.io.torch.load_checkpoint`

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

        if not (file_name.endswith(".pth") or file_name.endswith(".pt")):
            file_name = file_name + ".pt"

        return load_checkpoint_torch(file_name, **kwargs)

    def _update_state(self, new_state):
        """
        Update the state from a given new state

        Parameters
        ----------
        new_state : dict
            new state to update internal state from

        Returns
        -------
        :class:`PyTorchNetworkTrainer`
            the trainer with a modified state

        """

        if "model" in new_state:
            self.module.load_state_dict(new_state.pop("model"))

        if "optimizer" in new_state and new_state["optimizer"]:
            optim_state = new_state.pop("optimizer")
            for key in self.optimizers.keys():
                self.optimizers[key].load_state_dict(
                    optim_state[key])

        if "epoch" in new_state:
            self.start_epoch = new_state.pop("epoch")

        return super()._update_state(new_state)

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
            extensions = [".pt", ".pth"]
        return BaseNetworkTrainer._search_for_prev_state(path, extensions)
