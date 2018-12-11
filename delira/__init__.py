__version__ = '0.1'

import warnings
warnings.simplefilter('default', DeprecationWarning)

from .data_loading import BaseCacheDataset, BaseLazyDataset, BaseDataManager, \
    RandomSampler, SequentialSampler

from .logging import TrixiHandler, MultiStreamHandler
from .io import torch_load_checkpoint, torch_save_checkpoint

from .models import AbstractNetwork, AbstractPyTorchNetwork

__all__ = [
    'BaseCacheDataset',
    'BaseLazyDataset',
    'BaseDataManager',
    'RandomSampler',
    'SequentialSampler',
    'TrixiHandler',
    'MultiStreamHandler',
    'torch_save_checkpoint',
    'torch_load_checkpoint',
    'AbstractNetwork',
    'AbstractPyTorchNetwork'
]
