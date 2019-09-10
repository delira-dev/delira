import typing

from delira.models.backends.torchscript import AbstractTorchScriptNetwork

from delira.utils import DeliraConfig
from delira.training.backends.torch.experiment import PyTorchExperiment
from delira.training.backends.torch.utils import create_optims_default

from delira.training.backends.torchscript.trainer import \
    TorchScriptNetworkTrainer


class TorchScriptExperiment(PyTorchExperiment):
    def __init__(self,
                 config: typing.Union[str, DeliraConfig],
                 model_cls: AbstractTorchScriptNetwork,
                 n_epochs=None,
                 name=None,
                 save_path=None,
                 key_mapping=None,
                 val_score_key=None,
                 optim_builder=create_optims_default,
                 checkpoint_freq=1,
                 trainer_cls=TorchScriptNetworkTrainer,
                 **kwargs):
        """

        Parameters
        ----------
        config : :class:`DeliraConfig` or str
            the training config, if string is passed,
            it is treated as a path to a file, where the
            config is loaded from
        model_cls : Subclass of :class:`AbstractTorchScriptNetwork`
            the class implementing the model to train
        n_epochs : int or None
            the number of epochs to train, if None: can be specified later
            during actual training
        name : str or None
            the Experiment's name
        save_path : str or None
            the path to save the results and checkpoints to.
            if None: Current working directory will be used
        key_mapping : dict
            mapping between data_dict and model inputs (necessary for
            prediction with :class:`Predictor`-API), if no keymapping is
            given, a default key_mapping of {"x": "data"} will be used here
        val_score_key : str or None
            key defining which metric to use for validation (determining
            best model and scheduling lr); if None: No validation-based
            operations will be done (model might still get validated,
            but validation metrics can only be logged and not used further)
        optim_builder : function
            Function returning a dict of backend-specific optimizers.
            defaults to :func:`create_optims_default_pytorch`
        checkpoint_freq : int
            frequency of saving checkpoints (1 denotes saving every epoch,
            2 denotes saving every second epoch etc.); default: 1
        trainer_cls : subclass of :class:`TorchScriptNetworkTrainer`
            the trainer class to use for training the model, defaults to
            :class:`TorchScriptNetworkTrainer`
        **kwargs :
            additional keyword arguments

        """
        super().__init__(config=config, model_cls=model_cls,
                         n_epochs=n_epochs, name=name, save_path=save_path,
                         key_mapping=key_mapping,
                         val_score_key=val_score_key,
                         optim_builder=optim_builder,
                         checkpoint_freq=checkpoint_freq,
                         trainer_cls=trainer_cls,
                         **kwargs)
