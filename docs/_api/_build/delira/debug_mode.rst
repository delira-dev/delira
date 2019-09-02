def get_current_debug_mode():
    """
    Getter function for the current debug mode
    Returns
    -------
    bool
        current debug mode
    """
    return __DEBUG_MODE


def switch_debug_mode():
    """
    Alternates the current debug mode
    """
    set_debug_mode(not get_current_debug_mode())


def set_debug_mode(mode: bool):
    """
    Sets a new debug mode
    Parameters
    ----------
    mode : bool
        the new debug mode
    """
    global __DEBUG_MODE
    __DEBUG_MODE = mode

.. role:: hidden
    :class: hidden-section

.. currentmodule:: delira._debug_mode

Debug Mode
==========

Delira now contains a fully-fledged `Debug` mode, which disables all kinds of multiprocessing.

:hidden:`get_current_debug_mode`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: get_current_debug_mode

:hidden:`switch_debug_mode`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: switch_debug_mode

:hidden:`set_debug_mode`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: set_debug_mode
