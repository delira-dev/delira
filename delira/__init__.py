__version__ = '0.1.0'

import warnings
warnings.simplefilter('default', DeprecationWarning)
warnings.simplefilter('default', ModuleNotFoundError)

from .data_loading import BaseCacheDataset, BaseLazyDataset, BaseDataManager, \
    RandomSampler, SequentialSampler

from .logging import TrixiHandler, MultiStreamHandler

from .models import AbstractNetwork

__all__ = [
    'BaseCacheDataset',
    'BaseLazyDataset',
    'BaseDataManager',
    'RandomSampler',
    'SequentialSampler',
    'TrixiHandler',
    'MultiStreamHandler',
    'AbstractNetwork'
]

try:
    import torch
    from .io import torch_load_checkpoint, torch_save_checkpoint
    from .models import AbstractPyTorchNetwork
    from .data_loading import TorchvisionClassificationDataset

    __all__ += [
        'torch_save_checkpoint',
        'torch_load_checkpoint',
        'AbstractPyTorchNetwork',
        'TorchvisionClassificationDataset'
    ]

except ModuleNotFoundError as e:
    warnings.warn(e)
