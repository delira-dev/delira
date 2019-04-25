__version__ = '0.3.2'

# from .models import AbstractNetwork
# from .logging import TrixiHandler, MultiStreamHandler
# from .data_loading import BaseCacheDataset, BaseLazyDataset, BaseDataManager, \
#     RandomSampler, SequentialSampler
import json
import os
import warnings
warnings.simplefilter('default', DeprecationWarning)
warnings.simplefilter('ignore', ImportWarning)

# to register new pssible backends, they have to be added to this list.
# each backend should consist of a tuple of length 2 with the first entry
# being the package import name and the second being the backend abbreviation.
# E.g. TensorFlow's package is named 'tensorflow' but if the package is found,
# it will be considered as 'tf' later on
__POSSIBLE_BACKENDS = [("torch", "torch"), ("tensorflow", "tf")]
__BACKENDS = []


def _determine_backends():

    _config_file = __file__.replace("__init__.py", ".delira")
    # look for config file to determine backend
    # if file exists: load config into environment variables

    if not os.path.isfile(_config_file):
        _backends = {}
        # try to import all possible backends to determine valid backends

        import importlib
        for curr_backend in __POSSIBLE_BACKENDS:
            try:
                assert len(curr_backend) == 2
                assert all([isinstance(_tmp, str) for _tmp in curr_backend]), \
                    "All entries in current backend must be strings"

                # check if backend can be imported
                bcknd = importlib.util.find_spec(curr_backend[0])

                if bcknd is not None:
                    _backends[curr_backend[1]] = True
                else:
                    _backends[curr_backend[1]] = False
                del bcknd

            except ValueError:
                _backends[curr_backend[1]] = False

        with open(_config_file, "w") as f:
            json.dump({"version": __version__, "backend": _backends},
                      f, sort_keys=True, indent=4)

        del _backends

    # set values from config file to variable
    with open(_config_file) as f:
        _config_dict = json.load(f)
    for key, val in _config_dict.pop("backend").items():
        if val:
            __BACKENDS.append(key.upper())
    del _config_dict

    del _config_file


def get_backends():
    """
    Return List of currently available backends

    """

    if not __BACKENDS:
        _determine_backends()
    return __BACKENDS


# if "TORCH" in get_backends():
#     from .io import torch_load_checkpoint, torch_save_checkpoint
#     from .models import AbstractPyTorchNetwork
#     from .data_loading import TorchvisionClassificationDataset
