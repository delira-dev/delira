
# basic imports
from delira.data_loading.data_loader import DataLoader
from delira.data_loading.dataset import AbstractDataset, IterableDataset, \
    DictDataset, BaseCacheDataset, BaseExtendCacheDataset, BaseLazyDataset, \
    ConcatDataset, Nii3DCacheDatset, Nii3DLazyDataset
from delira.data_loading.augmenter import Augmenter
from delira.data_loading.data_manager import DataManager
from delira.data_loading.load_utils import LoadSample, LoadSampleLabel

from delira.data_loading.sampler import *


from delira import get_backends as _get_backends

# If torch backend is available: Import Torchvision dataset
if "TORCH" in _get_backends():
    from delira.data_loading.dataset import TorchvisionClassificationDataset


# if numba is installed: Import Numba Transforms
try:
    from delira.data_loading.numba_transform import NumbaTransform, \
        NumbaTransformWrapper, NumbaCompose
except ImportError:
    pass
