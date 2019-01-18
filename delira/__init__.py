__version__ = '0.2.1'

import warnings
warnings.simplefilter('default', DeprecationWarning)
warnings.simplefilter('ignore', ImportWarning)

import os
import json
_config_file = __file__.replace("__init__.py", ".delira")
# look for config file to determine backend
# if file exists: load config into environment variables

if not os.path.isfile(_config_file):
    _backends = {}
    # try to import backends to determine valid backends
    try:
        import torch
        _backends["torch"] = True
        del torch
    except ImportError:
        _backends["torch"] = False
    try:
        import tensorflow
        _backends["tf"] = True
        del tensorflow
    except ImportError:
        _backends["tf"] = False

    with open(_config_file, "w") as f:
        json.dump({ "version": __version__, "backend": _backends}, f, sort_keys=True, indent=4)

    del _backends

# set values from config file to environment variables
with open(_config_file) as f:
    _config_dict = json.load(f)
_backend_str = ""
for key, val in _config_dict.pop("backend").items():
    if val:
        _backend_str += "%s," % key
_config_dict["backend"] = _backend_str
for key, val in _config_dict.items():
    if isinstance(val, str):
        val = val.lower()
    os.environ["DELIRA_%s" % key.upper()] = val

del _backend_str
del _config_dict

del _config_file

from .data_loading import BaseCacheDataset, BaseLazyDataset, BaseDataManager, \
    RandomSampler, SequentialSampler

from .logging import TrixiHandler, MultiStreamHandler

from .models import AbstractNetwork

def get_backends():
    """
    Return List of current backends

    """
    return os.environ["DELIRA_BACKEND"].split(",")[:-1]

if "torch" in os.environ["DELIRA_BACKEND"]:
    from .io import torch_load_checkpoint, torch_save_checkpoint
    from .models import AbstractPyTorchNetwork
    from .data_loading import TorchvisionClassificationDataset
