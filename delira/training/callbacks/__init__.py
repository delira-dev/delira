from .abstract_callback import AbstractCallback
from .early_stopping import EarlyStopping

__all__ = [
    "AbstractCallback",
    "EarlyStopping"
]

try:
    from .pytorch_schedulers import DefaultPyTorchSchedulerCallback
    from .pytorch_schedulers import CosineAnnealingLRCallback as \
        CosineAnnealingLRCallbackPyTorch
    from .pytorch_schedulers import ExponentialLRCallback as \
        ExponentialLRCallbackPyTorch

    from .pytorch_schedulers import LambdaLRCallback as LambdaLRCallbackPyTorch
    from .pytorch_schedulers import MultiStepLRCallback as \
        MultiStepLRCallbackPyTorch
    from .pytorch_schedulers import ReduceLROnPlateauCallback as \
        ReduceLROnPlateauCallbackPyTorch
    from .pytorch_schedulers import StepLRCallback as StepLRCallbackPyTorch

    __all__ += [
        'DefaultPyTorchSchedulerCallback',
        'CosineAnnealingLRCallbackPyTorch',
        'ExponentialLRCallbackPyTorch',
        'LambdaLRCallbackPyTorch',
        'MultiStepLRCallbackPyTorch',
        'ReduceLROnPlateauCallbackPyTorch',
        'StepLRCallbackPyTorch'
    ]

except ImportError as e:
    import warnings
    warnings.warn(ImportWarning(e.msg))
