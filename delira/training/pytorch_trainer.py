import logging
import os
import warnings
from functools import partial

from batchgenerators.dataloading import MultiThreadedAugmenter

from delira import get_backends
from .base_trainer import BaseNetworkTrainer

logger = logging.getLogger(__name__)

if "TORCH" in get_backends():
    import torch
    from .train_utils import convert_torch_tensor_to_npy
    from .train_utils import create_optims_default_pytorch as \
        create_optims_default

    from ..io.torch import load_checkpoint, save_checkpoint
    from ..models import AbstractPyTorchNetwork

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
                     train_metrics=None,
                     val_metrics=None,
                     lr_scheduler_cls=None,
                     lr_scheduler_params=None,
                     gpu_ids=None,
                     save_freq=1,
                     optim_fn=create_optims_default,
                     logging_type="tensorboardx",
                     logging_kwargs=None,
                     fold=0,
                     callbacks=None,
                     start_epoch=1,
                     metric_keys=None,
                     convert_batch_to_npy_fn=convert_torch_tensor_to_npy,
                     mixed_precision=False,
                     mixed_precision_kwargs=None,
                     criterions=None,
                     val_freq=1,
                     ** kwargs):
            """

            Parameters
            ----------
            network : :class:`AbstractTfNetwork`
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
            train_metrics : dict, optional
                metrics, which will be evaluated during train phase
                (should work on framework's tensor types)
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
            val_freq : int
                validation frequency specifying how often to validate the
                trained model (a value of 1 denotes validating every epoch,
                a value of 2 denotes validating every second epoch etc.);
                defaults to 1
            **kwargs :
                additional keyword arguments

            """

            if optimizer_params is None:
                optimizer_params = {}
            if train_metrics is None:
                train_metrics = {}
            if val_metrics is None:
                val_metrics = {}
            if lr_scheduler_params is None:
                lr_scheduler_params = {}
            if gpu_ids is None:
                gpu_ids = []
            if logging_kwargs is None:
                logging_kwargs = {}
            if callbacks is None:
                callbacks = []
            if mixed_precision_kwargs is None:
                mixed_precision_kwargs = {"enable_caching": True,
                                          "verbose": False,
                                          "allow_banned": False}
            if (criterions is not None) ^ (losses is not None):
                if losses is not None:
                    crits = losses
                elif criterions is not None:
                    warnings.warn(DeprecationWarning(
                        "The 'criterions' argument is deprecated and will \
                         be removed in next release to unify APIs across \
                         backends. Use 'losses' instead "))
                    crits = criterions
            else:
                crits = losses
                warnings.warn(
                    RuntimeWarning("'criterions' and 'losses' have \
                                    been specified.Using the values in \
                                    'losses' since 'criterions' is deprecated \
                                    and will be removed"))

            super().__init__(
                network, save_path, crits, optimizer_cls, optimizer_params,
                train_metrics, val_metrics, lr_scheduler_cls,
                lr_scheduler_params, gpu_ids, save_freq, optim_fn, key_mapping,
                logging_type, logging_kwargs, fold, callbacks, start_epoch,
                metric_keys, convert_batch_to_npy_fn, val_freq)

            self._setup(network, optim_fn, optimizer_cls, optimizer_params,
                        lr_scheduler_cls, lr_scheduler_params, gpu_ids,
                        key_mapping, convert_batch_to_npy_fn,
                        mixed_precision, mixed_precision_kwargs)

            for key, val in kwargs.items():
                setattr(self, key, val)

        def _setup(self, network, optim_fn, optimizer_cls, optimizer_params,
                   lr_scheduler_cls, lr_scheduler_params, gpu_ids,
                   key_mapping, convert_batch_to_npy_fn, mixed_precision,
                   mixed_precision_kwargs):
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

            """

            self.optimizers = optim_fn(network, optimizer_cls,
                                       **optimizer_params)

            super()._setup(network, lr_scheduler_cls, lr_scheduler_params,
                           gpu_ids, key_mapping, convert_batch_to_npy_fn,
                           network.prepare_batch)

            try:
                from apex import amp
                self._amp_handle = amp.init(mixed_precision,
                                            *mixed_precision_kwargs)
                wrap_fn = self._amp_handle.wrap_optimizer

            except ImportError:
                if mixed_precision:
                    logger.warning("Apex was not found found, trying to \
                                    continue in full precision instead")
                from ..utils.context_managers import DefaultOptimWrapperTorch
                wrap_fn = DefaultOptimWrapperTorch

            # wrap optimizers by half_precision_optimizer via apex if
            # necessary
            self.optimizers = {k: wrap_fn(
                v, num_loss=len(self.losses)) for k, v
                in self.optimizers.items()}

            # Load latest epoch file if available
            if os.path.isdir(self.save_path):
                latest_state_path, latest_epoch = self._search_for_prev_state(
                    self.save_path, [".pt", ".pth"])

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

            if gpu_ids and torch.cuda.is_available():
                self.use_gpu = True
                if (len(gpu_ids) > 1) and (torch.cuda.device_count() > 1):
                    # use GPU 0 as default input GPU
                    self.input_device = torch.device("cuda:%d" % gpu_ids[0])

                    # Train on multiple GPUs and use GPU 0 as output device
                    self.module = torch.nn.DataParallel(self.module.to(
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

        def _at_training_end(self):
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

            return self.module

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
            datamgr : :class:`BaseDataManager`
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
            if metrics is None:
                metrics = {}

            self.module.eval()

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
            save_checkpoint(file_name, self.module, self.optimizers,
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
            :class:`BaseNetworkTrainer`
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
