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
            Function which scales the loss in ``apex`` and yields the unscaled loss 
            here to mirror the API

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
