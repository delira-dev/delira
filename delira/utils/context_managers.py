from delira import get_current_debug_mode, set_debug_mode


class DebugMode(object):
    """
    Context Manager to set a specific debug mode for the code inside the
    defined context (and reverting to previous mode afterwards)

    """

    def __init__(self, mode):
        """

        Parameters
        ----------
        mode : bool
            the debug mode; if ``True`` disables all multiprocessing
        """
        self._mode = mode

    def _switch_to_new_mode(self):
        """
        helper function to switch to the new debug mode
        (and saving the previous one in ``self._mode``)

        """
        prev_mode = get_current_debug_mode()
        set_debug_mode(self._mode)
        self._mode = prev_mode

    def __enter__(self):
        """
        Sets the specified debug mode on entering the context
        """
        self._switch_to_new_mode()

    def __exit__(self, *args, **kwargs):
        """
        Resets the previous debug mode on exiting the context

        Parameters
        ----------
        *args :
            arbitrary positional arguments
            (ignored here, just needed for compatibility with other context
            managers)
        **kwargs :
            arbitrary keyword arguments
            (ignored here, just needed for compatibility with other context
            managers)

        """
        self._switch_to_new_mode()


class DebugEnabled(DebugMode):
    """
    Context Manager to enable the debug mode for the wrapped context

    """

    def __init__(self):
        super().__init__(True)


class DebugDisabled(DebugMode):
    """
    Context Manager to disable the debug mode for the wrapped context
    """

    def __init__(self):
        super().__init__(False)
