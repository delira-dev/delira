import os
import logging
import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf
from batchgenerators.dataloading import MultiThreadedAugmenter
from .callbacks import AbstractCallback
from .abstract_trainer import AbstractNetworkTrainer
from .train_utils import create_optims_default_tf as create_optims_default
from .train_utils import initialize_uninitialized
from ..io import tf_load_checkpoint, tf_save_checkpoint
from delira.logging import TrixiHandler
from trixi.logger.tensorboard.tensorboardxlogger import TensorboardXLogger

logger = logging.getLogger(__name__)


class TfNetworkTrainer(AbstractNetworkTrainer):
    """
    Train and Validate a Network

    See Also
    --------
    :class:`AbstractNetwork`

    """

    def __init__(self,
                 network,
                 save_path,
                 losses: dict,
                 optimizer_cls,
                 optimizer_params={},
                 train_metrics={},
                 val_metrics={},
                 val_dataset_metrics={},
                 lr_scheduler_cls=None,
                 lr_scheduler_params={},
                 gpu_ids=[],
                 save_freq=1,
                 optim_fn=create_optims_default,
                 fold=0,
                 callbacks=[],
                 start_epoch=1,
                 metric_keys=None,
                 convert_batch_to_npy_fn=lambda x: x,
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
        val_dataset_metrics : dict, optional
            metrics, which will be evaluated during test phase on the whole 
            dataset (should work on numpy arrays)
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
            the identity function
        **kwargs :
            Additional keyword arguments

        """

        super().__init__(
            network, save_path, losses, optimizer_cls, optimizer_params,
            train_metrics, val_metrics, val_dataset_metrics, lr_scheduler_cls,
            lr_scheduler_params, gpu_ids, save_freq, optim_fn, fold, callbacks,
            start_epoch, metric_keys, convert_batch_to_npy_fn)

        # remove prior Trixihandlers and ensure logging of training results to self.save_path
        # This facilitates visualization of multiple splits/fold inside one tensorboard-instance by means of
        # different tf.Summary.FileWriters()
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            handler.close()
        root_logger.handlers = []
        logging.basicConfig(level=logging.INFO,
                            handlers=[TrixiHandler(TensorboardXLogger, 0, self.save_path)])

        self._setup(network, optim_fn, optimizer_cls, optimizer_params,
                    lr_scheduler_cls, lr_scheduler_params, gpu_ids)

        for key, val in kwargs.items():
            setattr(self, key, val)

    def _setup(self, network, optim_fn, optimizer_cls, optimizer_params,
               lr_scheduler_cls, lr_scheduler_params, gpu_ids):
        """
        Defines the Trainers Setup

        Parameters
        ----------
        network : instance of :class: `AbstractTfNetwork`
            the network to train
        optim_fn : function
            creates a dictionary containing all necessary optimizers
        optimizer_cls : subclass of tf.train.Optimizer
            optimizer class implementing the optimization algorithm of choice
        optimizer_params : dict
        lr_scheduler_cls : Any
            learning rate schedule class: must implement step() method
        lr_scheduler_params : dict
            keyword arguments passed to lr scheduler during construction
        gpu_ids : list
            list containing ids of GPUs to use; if empty: use cpu instead
        """

        # TODO: implement multi-GPU and single GPU training with help of
        #  https://www.tensorflow.org/api_docs/python/tf/keras/utils/multi_gpu_model
        #  note: might be bugged in combination with sess.run https://github.com/tensorflow/tensorflow/issues/21788
        #  and https://www.tensorflow.org/api_docs/python/tf/keras/models/clone_model

        """
        if gpu_ids and tf.test.is_gpu_available():
            assert len(gpu_ids) <= len(get_available_gpus()), "more GPUs specified than available"
            self.use_gpu = True
            if len(gpu_ids) > 1:
                logger.warning(
                    "multi-GPU training not yet tested!")

                network.model = tf.keras.utils.multi_gpu_model(
                                        network.model,
                                        len(gpu_ids),
                                        cpu_merge=True,
                                        cpu_relocation=False)
            else:
                network.models = tf.keras.models.clone_model(network.model)
        else:
            self.use_gpu = False
        """

        self.optimizers = optim_fn(optimizer_cls, **optimizer_params)

        super()._setup(network, lr_scheduler_cls, lr_scheduler_params, gpu_ids,
                       lambda x: x, lambda x: x)

        self.use_gpu = True

        self.module._add_losses(self.losses)
        self.module._add_optims(self.optimizers)
        # check for unitialized variables
        initialize_uninitialized(self.module._sess)

    def _at_training_end(self):
        """
        Defines Behaviour at end of training: Loads best model if available

        Returns
        -------
        :class:`AbstractTfNetwork`
            best network

        """
        if os.path.isfile(os.path.join(self.save_path, 'checkpoint_best.meta')):

            # load best model and return it. Since the state is hidden in the graph, we don't actually need to use
            # self.update_state.
            self.update_state(os.path.join(self.save_path,
                                           'checkpoint_best')
                              )

        return self.module

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
        self.module.training = True

        return super()._train_single_epoch(batchgen, epoch, verbose=verbose)

    def predict_data_mgr(self, datamgr, batch_size=None, verbose=False):
        """
        Returns predictions from network for batches from batchgen

        Parameters
        ----------
        batchgen : MultiThreadedAugmenter
            Generator yielding the batches to predict

        batch_size : None or int
            if int: collect batches until batch_size is reached and
            forward them together

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
        self.module.training = False

        return super().predict_data_mgr(datamgr, batch_size, verbose=verbose)

    def save_state(self, file_name, *args, **kwargs):
        """
        saves the current state via :func:`delira.io.tf.save_checkpoint`

        Parameters
        ----------
        file_name : str
            filename to save the state to
        """
        tf_save_checkpoint(file_name, self.module)

    def load_state(self, file_name, *args, **kwargs):
        """
        Loads the new state from file via :func:`delira.io.tf.load_checkpoint`

        Parameters
        ----------
        file_name : str
            the file to load the state from
        Returns
        -------

        """
        return tf_load_checkpoint(file_name, self.module)
