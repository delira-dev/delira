from delira import get_backends as _get_backends

if "TORCH" in _get_backends():
    from delira.training.backends.torchscript.experiment import \
        TorchScriptExperiment
    from delira.training.backends.torchscript.trainer import \
        TorchScriptNetworkTrainer
