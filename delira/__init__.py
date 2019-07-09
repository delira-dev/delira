__version__ = '0.4.1'

import warnings
warnings.simplefilter('default', DeprecationWarning)
warnings.simplefilter('ignore', ImportWarning)

from delira._backends import get_backends
from delira._debug_mode import get_current_debug_mode, switch_debug_mode, \
  set_debug_mode
