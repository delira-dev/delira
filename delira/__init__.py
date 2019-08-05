from delira._debug_mode import get_current_debug_mode, switch_debug_mode, \
    set_debug_mode
from delira._backends import get_backends, seed_all

from ._version import get_versions as _get_versions

import warnings
warnings.simplefilter('default', DeprecationWarning)
warnings.simplefilter('ignore', ImportWarning)


__version__ = _get_versions()['version']
del _get_versions
