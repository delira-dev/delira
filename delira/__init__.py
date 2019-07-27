from delira._debug_mode import get_current_debug_mode, switch_debug_mode, \
    set_debug_mode
from delira._backends import get_backends

from ._version import get_versions
import json
import os
import warnings
warnings.simplefilter('default', DeprecationWarning)
warnings.simplefilter('ignore', ImportWarning)

__version__ = get_versions()['version']
del get_versions
