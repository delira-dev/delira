from delira import get_backends
from delira.training.callbacks.abstract_callback import AbstractCallback

if 'TORCH' in get_backends():
    from torch.optim.lr_scheduler import ReduceLROnPlateau, \
        CosineAnnealingLR, ExponentialLR, LambdaLR, MultiStepLR, StepLR, \
        OneCycleLR

    class DefaultPyTorchSchedulerCallback(AbstractCallback):
        """
        Implements a Callback, which `at_epoch_end` function is suitable for
        most schedulers

        """

        def __init__(self, *args, **kwargs):
            """

            Parameters
            ----------
            *args :
                Arbitrary Positional Arguments
            **kwargs :
                Arbitrary Keyword Arguments

            """
            super().__init__()

            self.scheduler = None

        def at_epoch_end(self, trainer, **kwargs):
            """
            Executes a single scheduling step

            Parameters
            ----------
            trainer : :class:`PyTorchNetworkTrainer`
                the trainer class, which can be changed
            **kwargs :
                additional keyword arguments

            Returns
            -------
            :class:`PyTorchNetworkTrainer`
                modified trainer

            """
            self.scheduler.step(epoch=kwargs.get("curr_epoch", None))
            return {}

    class OneCycleLRCallback(DefaultPyTorchSchedulerCallback):
        """
        Wraps PyTorch's `OneCycleLR` Scheduler as Callback

        """

        def __init__(
                self,
                optimizer,
                max_lr,
                total_steps=None,
                epochs=None,
                steps_per_epoch=None,
                pct_start=0.3,
                anneal_strategy='cos',
                cycle_momentum=True,
                base_momentum=0.85,
                max_momentum=0.95,
                div_factor=25.0,
                final_div_factor=10000.0,
                last_epoch=-1):
            """

            Parameters
            ----------
            optimizer (Optimizer): Wrapped optimizer.
            max_lr (float or list): Upper learning rate boundaries in the cycle
                for each parameter group.
            total_steps (int): The total number of steps in the cycle. Note
                that if a value is provided here, then it must be inferred by
                providing a value for epochs and steps_per_epoch.
                Default: None
            epochs (int): The number of epochs to train for. This is used along
                with steps_per_epoch in order to infer the total number of
                steps in the cycle if a value for total_steps is not provided.
                Default: None
            steps_per_epoch (int): The number of steps per epoch to train for.
                This is used along with epochs in order to infer the total
                number of steps in the cycle if a value for total_steps is
                not provided.
                Default: None
            pct_start (float): The percentage of the cycle (in number of steps)
                spent increasing the learning rate.
                Default: 0.3
            anneal_strategy (str): {'cos', 'linear'}
                Specifies the annealing strategy.
                Default: 'cos'
            cycle_momentum (bool): If ``True``, momentum is cycled inversely
                to learning rate between 'base_momentum' and 'max_momentum'.
                Default: True
            base_momentum (float or list): Lower momentum boundaries in the
                cycle for each parameter group. Note that momentum is cycled
                inversely to learning rate; at the peak of a cycle, momentum is
                'base_momentum' and learning rate is 'max_lr'.
                Default: 0.85
            max_momentum (float or list): Upper momentum boundaries in the
                cycle for each parameter group. Functionally,
                it defines the cycle amplitude (max_momentum - base_momentum).
                Note that momentum is cycled inversely
                to learning rate; at the start of a cycle, momentum is
                'max_momentum' and learning rate is 'base_lr'
                Default: 0.95
            div_factor (float): Determines the initial learning rate via
                initial_lr = max_lr/div_factor
                Default: 25
            final_div_factor (float): Determines the minimum learning rate via
                min_lr = initial_lr/final_div_factor
                Default: 1e4
            last_epoch (int): The index of the last batch. This parameter is
                used when resuming a training job. Since `step()` should be
                invoked after each batch instead of after each epoch, this
                number represents the total number of *batches* computed,
                not the total number of epochs computed.
                When last_epoch=-1, the schedule is started from the
                beginning.
                Default: -1
            """
            super().__init__()
            self.scheduler = OneCycleLR(
                optimizer,
                max_lr,
                total_steps,
                epochs,
                steps_per_epoch,
                pct_start,
                anneal_strategy,
                cycle_momentum,
                base_momentum,
                max_momentum,
                div_factor,
                final_div_factor,
                last_epoch)

        def at_iter_begin(self, trainer, train,
                          **kwargs):
            """
            Executes a single scheduling step

            Parameters
            ----------
            trainer : :class:`PyTorchNetworkTrainer`
                the trainer class, which can be changed
            kwargs :
                additional keyword arguments

            Returns
            -------
            :class:`PyTorchNetworkTrainer`
                modified trainer

            """
            if train:
                self.scheduler.step()

            return {}

        def at_epoch_end(self, trainer, **kwargs):
            return {}

    class ReduceLROnPlateauCallback(DefaultPyTorchSchedulerCallback):
        """
        Wraps PyTorch's `ReduceLROnPlateau` Scheduler as Callback

        """

        def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                     verbose=False, threshold=1e-4, threshold_mode='rel',
                     cooldown=0, min_lr=0, eps=1e-8):
            """

            Parameters
            ----------
            optimizer : Optimizer
                Wrapped optimizer.
            mode : str
                One of `min`, `max`. In `min` mode, lr will
                be reduced when the quantity monitored has stopped
                decreasing; in `max` mode it will be reduced when the
                quantity monitored has stopped increasing. Default: 'min'.
            factor : float
                Factor by which the learning rate will be
                reduced. new_lr = lr * factor. Default: 0.1.
            patience : int
                Number of epochs with no improvement after
                which learning rate will be reduced. For example, if
                `patience = 2`, then we will ignore the first 2 epochs
                with no improvement, and will only decrease the LR after the
                3rd epoch if the loss still hasn't improved then.
                Default: 10.
            verbose : bool
                If ``True``, prints a message to stdout for
                each update. Default: ``False``.
            threshold : float
                Threshold for measuring the new optimum,
                to only focus on significant changes. Default: 1e-4.
            threshold_mode : string
                One of `rel`, `abs`. In `rel` mode,
                dynamic_threshold = best * ( 1 + threshold ) in 'max'
                mode or best * ( 1 - threshold ) in `min` mode.
                In `abs` mode, dynamic_threshold = best + threshold in
                `max` mode or best - threshold in `min` mode. Default: 'rel'.
            cooldown : int
                Number of epochs to wait before resuming
                normal operation after lr has been reduced. Default: 0.
            min_lr : float or list
                A scalar or a list of scalars. A
                lower bound on the learning rate of all param groups
                or each group respectively. Default: 0.
            eps : float
                Minimal decay applied to lr. If the difference
                between new and old lr is smaller than eps, the update is
                ignored. Default: 1e-8

            """
            super().__init__()
            self.scheduler = ReduceLROnPlateau(
                optimizer,
                mode,
                factor,
                patience,
                verbose,
                threshold,
                threshold_mode,
                cooldown,
                min_lr,
                eps)

        def at_epoch_end(self, trainer,
                         **kwargs):
            """
            Executes a single scheduling step

            Parameters
            ----------
            trainer : :class:`PyTorchNetworkTrainer`
                the trainer class, which can be changed
            kwargs :
                additional keyword arguments

            Returns
            -------
            :class:`PyTorchNetworkTrainer`
                modified trainer

            """
            val_metrics = kwargs.get("val_metrics", {})

            val_score_key = kwargs.get("val_score_key", None)

            metrics = val_metrics.get(val_score_key)

            self.scheduler.step(metrics=metrics)

            return {}

    class CosineAnnealingLRCallback(DefaultPyTorchSchedulerCallback):
        """
        Wraps PyTorch's `CosineAnnealingLR` Scheduler as callback

        """

        def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
            """

            Parameters
            ----------
            optimizer : optimizer
                Wrapped optimizer.
            T_max : int
                Maximum number of iterations.
            eta_min : float
                Minimum learning rate. Default: 0.
            last_epoch : int
                The index of last epoch. Default: -1.

            """
            super().__init__()

            self.scheduler = CosineAnnealingLR(optimizer, T_max, eta_min,
                                               last_epoch)

    class ExponentialLRCallback(DefaultPyTorchSchedulerCallback):
        """
        Wraps PyTorch's `ExponentialLR` Scheduler as callback

        """

        def __init__(self, optimizer, gamma, last_epoch=-1):
            """

            Parameters
            ----------
            optimizer : Optimizer
                Wrapped optimizer.
            gamma : float
                Multiplicative factor of learning rate decay.
            last_epoch : int
                The index of last epoch. Default: -1.

            """
            super().__init__()

            self.scheduler = ExponentialLR(optimizer, gamma, last_epoch)

    class LambdaLRCallback(DefaultPyTorchSchedulerCallback):
        """
        Wraps PyTorch's `LambdaLR` Scheduler as callback

        """

        def __init__(self, optimizer, lr_lambda, last_epoch=-1):
            """

            Parameters
            ----------
            optimizer : Optimizer
                Wrapped optimizer.
            lr_lambda : function or list
                A function which computes a multiplicative
                factor given an integer parameter epoch, or a list of such
                functions, one for each group in optimizer.param_groups.
            last_epoch : int
                The index of last epoch. Default: -1.

            """
            super().__init__()

            self.scheduler = LambdaLR(optimizer, lr_lambda, last_epoch)

    class MultiStepLRCallback(DefaultPyTorchSchedulerCallback):
        """
        Wraps PyTorch's `MultiStepLR` Scheduler as callback

        """

        def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
            """

            Parameters
            ----------
            optimizer : Optimizer
                Wrapped optimizer.
            milestones : list
                List of epoch indices. Must be increasing.
            gamma : float
                Multiplicative factor of learning rate decay.
                Default: 0.1.
            last_epoch : int
                The index of last epoch. Default: -1.

            """
            super().__init__()

            self.scheduler = MultiStepLR(
                optimizer, milestones, gamma, last_epoch)

    class StepLRCallback(DefaultPyTorchSchedulerCallback):
        """
        Wraps PyTorch's `StepLR` Scheduler as callback

        """

        def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
            """

            Parameters
            ----------
            optimizer : Optimizer
                Wrapped optimizer.
            step_size : int
                Period of learning rate decay.
            gamma :float
                Multiplicative factor of learning rate decay.
                Default: 0.1.
            last_epoch : int
                The index of last epoch. Default: -1

            """
            super().__init__()

            self.scheduler = StepLR(optimizer, step_size, gamma, last_epoch)
