from delira.training.backends.tf_graph.utils import initialize_uninitialized
from delira.training.backends.tf_eager.utils import create_optims_default
from delira.training.utils import convert_to_numpy_identity \
    as convert_to_numpy
from delira.training.base_trainer import BaseNetworkTrainer
from delira.training.callbacks.logging_callback import DefaultLoggingCallback
from delira.io.tf import load_checkpoint, save_checkpoint
from delira.models.backends.tf_graph import AbstractTfGraphNetwork
from delira.data_loading import DataManager
import os
import logging

from tensorflow import executing_eagerly

from batchgenerators.dataloading import MultiThreadedAugmenter

logger = logging.getLogger(__name__)


class TfGraphNetworkTrainer(BaseNetworkTrainer):
    """
    Train and Validate a Network

    See Also
    --------
    :class:`AbstractNetwork`

    """

    def __init__(self,
                 network: AbstractTfGraphNetwork,
                 save_path: str,
                 key_mapping: dict,
                 losses: dict,
                 optimizer_cls,
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
                 val_freq=1,
                 **kwargs
                 ):
        """

        Parameters
        ----------
        network : :class:`AbstractTfGraphNetwork`
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
                'last' | 'first' | 'mean' 'median' | 'max' | 'min'.
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
        val_freq : int
            validation frequency specifying how often to validate the trained
            model (a value of 1 denotes validating every epoch,
            a value of 2 denotes validating every second epoch etc.);
            defaults to 1
        **kwargs :
            Additional keyword arguments

        """
        assert not executing_eagerly()

        if optimizer_params is None:
            optimizer_params = {}
        if metrics is None:
            metrics = {}
        if lr_scheduler_params is None:
            lr_scheduler_params = {}
        if gpu_ids is None:
            gpu_ids = []
        if logging_kwargs is None:
            logging_kwargs = {}
        if callbacks is None:
            callbacks = []

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
                    lr_scheduler_cls, lr_scheduler_params,
                    key_mapping, convert_batch_to_npy_fn, gpu_ids, callbacks)

        for key, val in kwargs.items():
            setattr(self, key, val)

    def _setup(self, network, optim_fn, optimizer_cls, optimizer_params,
               lr_scheduler_cls, lr_scheduler_params, key_mapping,
               convert_batch_to_npy_fn, gpu_ids, callbacks):
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
        convert_batch_to_npy_fn : type, optional
            function converting a batch-tensor to numpy, per default this is
            the identity function
        gpu_ids : list
            list containing ids of GPUs to use; if empty: use cpu instead
        callbacks : list
            initial callbacks to register

        Raises
        ------
        RuntimeError
            if multiple GPU ids passed
        """

        # TODO: implement multi-GPU and single GPU training with help of
        #  keras multi-gpu model
        #  note: might be bugged in combination with sess.run
        #  https://github.com/tensorflow/tensorflow/issues/21788

        # if gpu_ids and tf.test.is_gpu_available():
        #     assert len(gpu_ids) <= len(get_available_gpus()), "more GPUs
        #     specified than available"
        #     self.use_gpu = True
        #     if len(gpu_ids) > 1:
        #         logger.warning(
        #             "multi-GPU training not yet tested!")
        #
        #         network.model = tf.keras.utils.multi_gpu_model(
        #                                 network.model,
        #                                 len(gpu_ids),
        #                                 cpu_merge=True,
        #                                 cpu_relocation=False)
        #     else:
        #         network.models = tf.keras.models.clone_model(network.model)
        # else:
        #     self.use_gpu = False
        #
        if len(gpu_ids) > 1:
            raise RuntimeError("Multiple GPUs not yet supported")

        self.optimizers = optim_fn(optimizer_cls, **optimizer_params)

        super()._setup(network, lr_scheduler_cls, lr_scheduler_params, gpu_ids,
                       key_mapping, convert_batch_to_npy_fn, lambda x: x,
                       callbacks)

        self.use_gpu = True

        self.module._add_losses(self.losses)
        self.module._add_optims(self.optimizers)
        # check for unitialized variables
        initialize_uninitialized(self.module._sess)

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

                self.update_state(latest_state_path)
                self.start_epoch = latest_epoch

    def _at_training_end(self, *args, **kwargs):
        """
        Defines Behaviour at end of training: Loads best model if available

        Returns
        -------
        :class:`AbstractTfNetwork`
            best network

        """

        if os.path.isfile(os.path.join(self.save_path,
                                       'checkpoint_best.meta')):

            # load best model and return it.

            self.update_state(os.path.join(self.save_path,
                                           'checkpoint_best')
                              )

        return super()._at_training_end(*args, **kwargs)

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
        self.module.training = True

        return super()._train_single_epoch(dmgr_train, epoch, verbose=verbose)

    def predict_data_mgr(self, datamgr, batch_size=None, metrics=None,
                         metric_keys=None, verbose=False, **kwargs):
        """
        Defines a routine to predict data obtained from a batchgenerator

        Parameters
        ----------
        datamgr : :class:`DataManager`
            Manager producing a generator holding the batches
        batch_size : int
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

        """
        if metrics is None:
            metrics = {}

        self.module.training = False

        return super().predict_data_mgr(datamgr, batch_size, metrics,
                                        metric_keys, verbose=verbose)

    def save_state(self, file_name, *args, **kwargs):
        """
        saves the current state via :func:`delira.io.tf.save_checkpoint`

        Parameters
        ----------
        file_name : str
            filename to save the state to
        """
        save_checkpoint(file_name, self.module)

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
        return load_checkpoint(file_name, self.module)

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
            if no latest checkpoint was found
        int
            the latest epoch (1 if no checkpoint was found)

        """
        if extensions is None:
            extensions = [".meta"]
        return BaseNetworkTrainer._search_for_prev_state(path, extensions)
