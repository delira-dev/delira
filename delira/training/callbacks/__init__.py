from delira import get_backends

from delira.training.callbacks.logging_callback import DefaultLoggingCallback
from delira.training.callbacks.abstract_callback import AbstractCallback
from delira.training.callbacks.early_stopping import EarlyStopping

if "TORCH" in get_backends():
    from delira.training.callbacks.pytorch_schedulers import \
        DefaultPyTorchSchedulerCallback
    from delira.training.callbacks.pytorch_schedulers import \
        CosineAnnealingLRCallback as CosineAnnealingLRCallbackPyTorch
    from delira.training.callbacks.pytorch_schedulers import \
        ExponentialLRCallback as ExponentialLRCallbackPyTorch

    from delira.training.callbacks.pytorch_schedulers import \
        LambdaLRCallback as LambdaLRCallbackPyTorch
    from delira.training.callbacks.pytorch_schedulers import \
        MultiStepLRCallback as MultiStepLRCallbackPyTorch
    from delira.training.callbacks.pytorch_schedulers import \
        ReduceLROnPlateauCallback as ReduceLROnPlateauCallbackPyTorch
    from delira.training.callbacks.pytorch_schedulers import StepLRCallback \
        as StepLRCallbackPyTorch
    from delira.training.callbacks.pytorch_schedulers import \
        OneCycleLRCallback as OneCycleLRCallbackPyTorch
