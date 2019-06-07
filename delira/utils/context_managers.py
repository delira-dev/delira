import contextlib

from delira import get_backends

if "TORCH" in get_backends():
    import torch

    class DefaultOptimWrapperTorch(object):
        """
        Class wrapping a ``torch`` optimizer to mirror the behavior of ``apex``
        without depending on it

        """

        def __init__(self, optimizer: torch.optim.Optimizer, *args, **kwargs):
            """

            Parameters
            ----------
            optimizer : torch.optim.Optimizer
                the actual optimizer to wrap
            *args :
                additional positional arguments (unused)
            **kwargs :
                additional keyword arguments (unused)

            """

            self._optimizer = optimizer

        @contextlib.contextmanager
        def scale_loss(self, loss):
            """
            Function which scales the loss in ``apex`` and yields the unscaled
            loss here to mirror the API

            Parameters
            ----------
            loss : torch.Tensor
                the unscaled loss

            """

            yield loss
            return

        def step(self, closure=None):
            """
            Wraps the step method of the optimizer and calls the original step
            method

            Parameters
            ----------
            closure : callable
                A closure that reevaluates the model and returns the loss.
                Optional for most optimizers.

            """

            return self._optimizer.step(closure=closure)

        # Forward any attribute lookups
        def __getattr__(self, attr):
            return getattr(self._optimizer, attr)

        # Forward all torch.optim.Optimizer methods
        def __getstate__(self):
            return self._optimizer.__getstate__()

        def __setstate__(self, *args, **kwargs):
            return self._optimizer.__setstate__(*args, **kwargs)

        def __repr__(self):
            return self._optimizer.__repr__()

        def state_dict(self):
            return self._optimizer.state_dict()

        def load_state_dict(self, state_dict):
            return self._optimizer.load_state_dict(state_dict)

        def zero_grad(self):
            return self._optimizer.zero_grad()

        def add_param_group(self, param_group):
            return self._optimizer.add_param_group(param_group)

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
