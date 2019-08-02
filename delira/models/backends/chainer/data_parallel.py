from delira.models.backends.chainer.abstract_network import \
    AbstractChainerNetwork
import chainer


def _apply_scatter(inputs: chainer.Variable, target_devices: list,
                   dim: int = 0):
    """
    Scatters inputs to target devices; Slicing will be done against a given
    dimension

    Parameters
    ----------
    inputs : :class:`chainer.Variable`
        the input variable to scatter
    target_devices : list
        the target devices to scatter to
    dim : int
        the dimension to use for slicing

    Returns
    -------
    list
        list of variable slices on correct devices

    """

    def _slice_inputs(input_var, dim, num_dims, start, end, target_device):
        """
        Slices the input variable along a given dimension from start to end
        and pushes it to correct device

        Parameters
        ----------
        input_var : :class:`chainer.Variable`
            the variable to slice
        dim : int
            the dimension to slice along
        num_dims : int
            the dimensionality of ``input_var``
        start : int
            the start value for slicing (included)
        end : int
            the end value for slicing (excluded)
        target_device: str or :class:`chainer.backend.Device`
            the device to push to

        Returns
        -------
        :class:`chainer.Variable`
            the slice of the variable

        """
        slc = [slice(None)] * num_dims
        slc[dim] = slice(start, end)
        sliced_var = input_var[slc]
        sliced_var.to_device(target_device)
        output_shape = list(input_var.shape)
        output_shape[dim] = -1
        return sliced_var.reshape(output_shape)

    # create empty sliced input list
    scattered_inputs = []

    # calculate constant only once
    num_devices = len(target_devices)
    samples_per_device = inputs.shape[dim] // num_devices
    num_dims = len(inputs.shape)

    # iterate over number of devices and slice accordingly
    # (exclude last device)
    # iterating until the minimum of num_devices and inputs.shape[dim] -1
    # ensures that if the batchsize is too small to be scattered across all
    # devices, we will only scatter across as many devices as possible
    for i in range(min(num_devices, inputs.shape[dim]) - 1):
        start, end = i * samples_per_device, i + 1 * samples_per_device
        scattered_inputs.append(_slice_inputs(inputs, dim,
                                              num_dims, start, end,
                                              target_devices[i]))

    # all remaining samples (not yet sliced) are appended now
    # (all samples used; will be pushed to last device later)
    scattered_inputs.append(_slice_inputs(
        inputs, dim, len(inputs.shape,),
        (num_devices - 1) * samples_per_device,
        inputs.shape[dim], target_devices[-1]))

    return scattered_inputs


def _apply_gather(target_device, dim, *outputs):
    for _output in outputs:
        _output.to_device(target_device)

    return chainer.functions.concat(outputs, dim)


def _scatter(inputs, target_devices: list, dim):
    """
    Scatters all inputs across given target_devices

    Parameters
    ----------
    inputs : Any
    target_devices : list
        list of devices to scatter to
    dim : int
        dimension to use for slicing

    Returns
    -------
    list
        list of scattered inputs

    """

    def _scatter_map(inputs):
        """
        Scatters all inputs across given target_devices

        Parameters
        ----------
        inputs : Any

        Returns
        -------
        list
            list of scattered inputs

        """

        # directly apply the scattering on variable
        if isinstance(inputs, chainer.Variable):
            return _apply_scatter(inputs, target_devices, dim)

        # map _scatter_map recursively to all samples in tuple
        if isinstance(inputs, tuple) and inputs:
            return list(zip(*map(_scatter_map, inputs)))

        # map _scatter_map recursively to all samples in list
        if isinstance(inputs, list) and inputs:
            return list(map(list, zip(*map(_scatter_map,
                                           inputs))))

        # map _scatter_map recursively to all samples in dict
        if isinstance(inputs, dict) and inputs:
            return list(map(type(inputs), zip(*map(_scatter_map,
                                                   inputs.items()))))

        # try to convert inputs to chainer variable first and afterwards
        # apply _scatter_map again

        try:
            return _scatter_map(chainer.as_variable(inputs))
        except TypeError:
            return [inputs for targets in target_devices]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has
    # references to a closure that has a reference to the scatter_map cell
    # (because the fn is recursive). To avoid this reference cycle, we set
    # the function to None, clearing the cell

    try:
        return _scatter_map(inputs)
    finally:
        _scatter_map = None


def _gather(outputs, target_device, dim=0):
    r"""
    Gathers tensors from different GPUs on a specified device
      (-1 means the CPU).
    """

    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, chainer.Variable):
            return _apply_gather(target_device, dim, *outputs)
        if out is None:
            return None
        if isinstance(out, dict):
            if not all((len(out) == len(d) for d in outputs)):
                raise ValueError(
                    'All dicts must have the same number of keys')

            return type(out)(((k, gather_map([d[k] for d in outputs]))
                              for k in out))
        return type(out)(map(gather_map, zip(*outputs)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        return gather_map(outputs)
    finally:
        gather_map = None


class DataParallelChainerNetwork(AbstractChainerNetwork):
    """
    A Wrapper around a :class:`AbstractChainerNetwork` instance to implement
    parallel training by splitting the batches
    """

    def __init__(self, module: AbstractChainerNetwork, devices: list,
                 output_device=None,
                 batch_dim=0):
        """

        Parameters
        ----------
        module : :class:`AbstractChainerNetwork`
            the module to wrap (will be replicated on all devices)
        devices : list
            a list containing the devices to use (either as strings or as
            :class:`chainer.backend.Device`).
        output_device : str or :class:`chainer.backend.Device`
            The output device
            Make sure, your labels are also on this device
            for loss calculation!
            If not specified, the second device of ``devices`` will be used
            for output gathering.
        batch_dim : int
            the index of the batchdimension (usually 0, but can become
            e.g. 1 in NLP tasks)

        """
        super().__init__()

        modules = [module.copy() for _ in devices]

        for _module, _device in zip(modules, devices):
            _module.to_device(_device)

        with self.init_scope():
            self.modules = chainer.ChainList(*modules)

        self.devices = devices

        if output_device is None:
            output_device = devices[1]

        self._output_device = output_device
        assert self._output_device in self.devices
        self._output_device_idx = self.devices.index(self._output_device)
        self.dim = batch_dim

    def forward(self, *args, **kwargs):
        """
        Scatters the inputs (both positional and keyword arguments) across
        all devices, feeds them through model replicas and re-builds
        batches on output device

        Parameters
        ----------
        *args :
            positional arguments of arbitrary number and type
        **kwargs :
            keyword arguments of arbitrary number and type

        Returns
        -------
        Any
            combined output from all scattered models

        """
        scattered_args, scattered_kwargs = self._scatter(args, kwargs,
                                                         self.devices,
                                                         self.dim)
        predictions = []

        for _args, _kwargs, _module in zip(scattered_args,
                                           scattered_kwargs,
                                           self.modules):

            predictions.append(_module(*_args, **_kwargs))

        predictions = self._gather(predictions, self.dim,
                                   self._output_device)

        return predictions

    def params(self, include_uninit=True):
        """
        Only the parameters of the module on the first device will actually
        be updated, all the other parameters will be replicated by the
        optimizer after an update

        Parameters
        ----------
        include_uninit : bool

        Returns
        -------
        a generator holding the root-modules parameters
        """
        return self.modules[0].params(include_uninit)

    @staticmethod
    def _scatter(inputs, kwargs, target_devices: list, dim=0):
        """
        Scatters all inputs (args and kwargs) to target devices and splits
        along given dimension

        Parameters
        ----------
        inputs : list or tuple
            positional arguments
        kwargs : dict
            keyword arguments
        target_devices : list
            list of target device (either string or chainer.backend.Device)
        dim : int
            the dimension, which should be used for splitting the batch

        Returns
        -------
        tuple
            scattered positional arguments
        tuple
            scattered keyword arguments

        """

        # scatter inputs if given
        inputs = _scatter(inputs, target_devices, dim) if inputs else []
        # scatter kwargs if given
        kwargs = _scatter(kwargs, target_devices, dim) if kwargs else []

        # extend lengths by empty tuples if necessary
        if len(inputs) < len(kwargs):
            inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
        elif len(kwargs) < len(inputs):
            kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])

        inputs = tuple(inputs)
        kwargs = tuple(kwargs)

        return inputs, kwargs

    @staticmethod
    def _gather(predictions, dim, target_device):
        """
        Re-Builds batches on the target device

        Parameters
        ----------
        predictions : list
            list containing the predictions from all replicated models
        dim : int
            dimension to use for concatenating single predictions
        target_device : str or chainer.backend.Device
            the device, the re-built batch should lie on

        Returns
        -------
        Any
            the rebuild batch (lying on ``target_device``)

        """
        return _gather(predictions, target_device, dim)

    def cleargrads(self):
        for module in self.modules:
            module.cleargrads()

    def zerograds(self):
        for module in self.modules:
            module.zerograds()

    @property
    def closure(self):
        return self.modules[0].closure

    @property
    def prepare_batch(self):
        return self.modules[0].prepare_batch


class ParallelOptimizerCumulateGradientsHook(object):
    """
    A hook which sums up all replication's gradients in a
    DataParallel-Scenario
    """

    name = "DataParallelCumulateGradients"
    call_for_each_param = False
    timing = 'pre'

    def __call__(self, optimizer: chainer.Optimizer):
        """
        Summing up all parameters if the target is an instance of
        ``DataParallel``

        Parameters
        ----------
        optimizer : chainer.Optimizer
            the optimizer holding the target, whoose gradients should be
            summed across the replications

        """
        if isinstance(optimizer.target, DataParallelChainerNetwork):
            for module in optimizer.target.modules[1:]:
                optimizer.target.modules[0].addgrads(module)


class ParallelOptimizerUpdateModelParameters(object):
    """
    A hook to replicate all parameters from the root model, to all
    model-replicas after the optimizer step
    """

    name = "DataParallelUpdateModelParams"
    call_for_each_param = False
    timing = "post"

    def __call__(self, optimizer: chainer.Optimizer):
        if isinstance(optimizer.target, DataParallelChainerNetwork):
            for module in optimizer.target.modules[1:]:
                module.copyparams(optimizer.target.modules[0])


class DataParallelChainerOptimizer(chainer.Optimizer):
    """
    An Optimizer-Wrapper to enable DataParallel. Basically this forwards
    all functions to the interal optimizer, but registers the additional
    hooks needed for DataParallel (namely
    :class:`ParallelOptimizerUpdateModelParameters` as a post-update hook
    and :class:`ParallelOptimizerCumulateGradientsHook` as a pre-update hook)

    """

    def __init__(self, optimizer):
        """

        Parameters
        ----------
        optimizer : :class:`chainer.Optimizer`
            the optimizer to wrap

        """
        if isinstance(optimizer, chainer.Optimizer):
            self._optimizer = optimizer

        else:
            raise RuntimeError("Invalid optimizer class given: Expected "
                               "instance of chainer.Optimizer, but got %s"
                               % optimizer.__class__.__name__)

    @classmethod
    def from_optimizer_class(cls, optim_cls, *args, **kwargs):
        """

        Parameters
        ----------
        optim_cls : subclass of :class:`chainer.Optimizer`
            the optimizer to use internally
        *args :
            arbitrary positional arguments (will be used for
            initialization of internally used optimizer)
        **kwargs :
            arbitrary keyword arguments (will be used for initialization
            of internally used optimizer)

        """
        if optim_cls is not None and issubclass(optim_cls,
                                                chainer.Optimizer):
            _optim = optim_cls(*args, **kwargs)
        else:
            raise RuntimeError("Invalid optimizer class given: Expected "
                               "Subclass of chainer.Optimizer, but got %s"
                               % optim_cls.__name__)
        return cls(_optim)

    def setup(self, link):
        """
        Calls the setup method of the internal optimizer and registers the
        necessary grads for data-parallel behavior

        Parameters
        ----------
        link : :class:`DataParallel`
            the target, whoose parameters should be updated

        """
        self._optimizer.setup(link)

        self._optimizer.add_hook(ParallelOptimizerCumulateGradientsHook())
        self._optimizer.add_hook(ParallelOptimizerUpdateModelParameters())

    @property
    def target(self):
        return self._optimizer.target

    @property
    def epoch(self):
        return self._optimizer.epoch

    @property
    def _pre_update_hooks(self):
        return self._optimizer._pre_update_hooks

    @property
    def _loss_scale(self):
        return self._optimizer._loss_scale

    @property
    def _loss_scale_max(self):
        return self._optimizer._loss_scale_max

    @property
    def _loss_scaling_is_dynamic(self):
        return self._optimizer._loss_scaling_is_dynamic

    @property
    def use_auto_new_epoch(self):
        return self._optimizer.use_auto_new_epoch

    @property
    def update(self):
        return self._optimizer.update

    @property
    def new_epoch(self):
        return self._optimizer.new_epoch

    @property
    def add_hook(self):
        return self._optimizer.add_hook

    @property
    def remove_hook(self):
        return self._optimizer.remove_hook

    @property
    def call_hooks(self):
        return self._optimizer.call_hooks

    @property
    def serialize(self):
        return self._optimizer.serialize

    @property
    def loss_scaling(self):
        return self._optimizer.loss_scaling

    @property
    def set_loss_scale(self):
        return self._optimizer.set_loss_scale

    @property
    def check_nan_in_grads(self):
        return self._optimizer.check_nan_in_grads

    @property
    def is_safe_to_update(self):
        return self._optimizer.is_safe_to_update

    @property
    def update_loss_scale(self):
        return self._optimizer.update_loss_scale
