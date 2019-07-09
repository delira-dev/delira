from delira._debug_mode import get_current_debug_mode, switch_debug_mode, \
    set_debug_mode
from delira._backends import get_backends
import warnings
warnings.simplefilter('default', DeprecationWarning)
warnings.simplefilter('ignore', ImportWarning)

from delira._version import __version__
