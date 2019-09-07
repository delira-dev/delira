import logging

from delira.io.torch import load_checkpoint_torchscript, \
    save_checkpoint_torchscript
from delira.models.backends.torchscript import AbstractTorchScriptNetwork

from delira.training.base_trainer import BaseNetworkTrainer
from delira.training.backends.torch.trainer import PyTorchNetworkTrainer

from delira.training.backends.torch.utils import convert_to_numpy
from delira.training.backends.torch.utils import create_optims_default

from delira.training.callbacks.logging_callback import DefaultLoggingCallback


logger = logging.getLogger(__name__)


class TorchScriptNetworkTrainer(PyTorchNetworkTrainer):
    def __init__(self,
                 network: AbstractTorchScriptNetwork,
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
                 fold=0,
                 callbacks=None,
                 start_epoch=1,
                 metric_keys=None,
                 convert_batch_to_npy_fn=convert_to_numpy,
                 criterions=None,
                 val_freq=1,
                 logging_callback_cls=DefaultLoggingCallback,
                 logging_frequencies=None,
                 logging_reduce_types=None,
                 **kwargs):
        """

        Parameters
        ----------
        network : :class:`AbstractPyTorchJITNetwork`
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
            Currently ``torch.jit`` only supports single GPU-Training,
            thus only the first GPU will be used if multiple GPUs are
            passed
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
            as target; default: None, which will result in key "label" for
            all metrics
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
            trained
            model (a value of 1 denotes validating every epoch,
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

        if len(gpu_ids) > 1:
            # only use first GPU due to
            # https://github.com/pytorch/pytorch/issues/15421
            gpu_ids = [gpu_ids[0]]
            logging.warning("Multiple GPUs specified. Torch JIT currently "
                            "supports only single-GPU training. "
                            "Switching to use only the first GPU "
                            "for now...")

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
                         mixed_precision=False,
                         mixed_precision_kwargs={},
                         **kwargs
                         )

    def save_state(self, file_name, epoch, **kwargs):
        """
        saves the current state via
        :func:`delira.io.torch.save_checkpoint_jit`

        Parameters
        ----------
        file_name : str
            filename to save the state to
        epoch : int
            current epoch (will be saved for mapping back)
        **kwargs :
            keyword arguments

        """
        if file_name.endswith(".ptj"):
            file_name = file_name.rsplit(".", 1)[0]

        save_checkpoint_torchscript(file_name, self.module,
                                    self.optimizers, **kwargs)

    @staticmethod
    def load_state(file_name, **kwargs):
        """
        Loads the new state from file via
        :func:`delira.io.torch.load_checkpoint:jit`

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
        return load_checkpoint_torchscript(file_name, **kwargs)

    def _update_state(self, new_state):
        """
        Update the state from a given new state

        Parameters
        ----------
        new_state : dict
            new state to update internal state from

        Returns
        -------
        :class:`TorchScriptNetworkTrainer`
            the trainer with a modified state

        """
        if "model" in new_state:
            self.module = new_state.pop("model").to(self.input_device)

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
            extensions = [".model.ptj"]
        return BaseNetworkTrainer._search_for_prev_state(path, extensions)
