
How To: Integrate your own Computation Backend
==============================================

*Author: Justus Schock*

*Date: 15.05.2019*

This howto will take you on a trip through the ``delira`` internals,
while we will see, how to add a custom computation backend on the
examplaric case of the ``torch.jit`` or ``TorchScript`` backend)

Model Definitions
-----------------

In order to implement a network, we will first have to define the
network itself. In ``delira`` there is a single backend-specific
implementation of an abstract network class for each of the backends.
These interface classes are all based on the ``AbstractNetwork``-class,
defining the major API.

So let's start having a look at this class to see, what we will have to
implement for our own backend.

Of course we will have to implement an ``__init__`` defining our class.
The ``__init__`` of ``AbstractNetwork`` (which should be called during
our the ``__init__`` of our baseclass) accepts a number of kwargs and
simply registers them to be ``init_kwargs``, so there is nothing we have
to take care of.

The next function to inspect is the ``__call__`` function, which makes
the class callable and the docstrings indicate, that it should take care
of our model's forward-pass.

After the ``__call__`` we now have the ``closure`` function, which
defines a single training step (including, but not limited to,
forward-pass, calculation of losses and train-metrics, backward-pass and
optimization).

The last method to implement is the ``prepare_batch`` function which
converts the input to a suitable format and the correct data-type and
device.

TorchScript Limitations
~~~~~~~~~~~~~~~~~~~~~~~

Since we want to implement an abstract network class for this specific
backend, we should have a look on how to generally implement models in
this backend.

According the the `PyTorch
docs <https://pytorch.org/docs/stable/jit.html>`__ this works as
follows:

    You can write TorchScript code directly using Python syntax. You do
    this using the ``torch.jit.script`` decorator (for functions) or
    ``torch.jit.script_method`` decorator (for methods) on subclasses of
    ``ScriptModule``. With this decorator the body of the annotated
    function is directly translated into TorchScript. TorchScript itself
    is a subset of the Python language, so not all features in Python
    work, but we provide enough functionality to compute on tensors and
    do control-dependent operations.

Since our use-case is to implement the interface class for networks, we
want to use the way of subclassing ``torch.jit.ScriptModule``, implement
it's ``forward`` and use the ``torch.jit.script_method`` decorator on
it.

The example given in the very same docs for this case is:

.. code:: ipython3

    import torch
    class MyScriptModule(torch.jit.ScriptModule):
        def __init__(self, N, M):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.rand(N, M))
    
        @torch.jit.script_method
        def forward(self, input):
            return self.weight.mv(input)
        
    my_script_module = MyScriptModule(5, 3)
    input_tensor = torch.rand(3)
    my_script_module(input_tensor)




.. parsed-literal::

    tensor([0.4997, 0.2955, 0.1588, 0.1873, 0.4753], grad_fn=<MvBackward>)



Merging TorchScript into our Abstract Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This little example gives us a few things, we have to do for a
successful definition of our base class:

**1.)** Our class has to subclass both, the ``AbstractNetwork`` and the
``torch.jit.ScriptModule`` classes.

**2.)** We need to implement a ``forward`` method, which takes care of
the forward-pass (as it's name indicates).

**3.)** We don't have to take care of the backward-pass (thanks to
``PyTorch``'s and ``TorchScript``'s AutoGrad (which is a framework for
automatic differentiation).

**4.)** Since ``torch.jit.ScriptModule`` is callable (seen in the
example), it already implements a ``__call__`` method and we may simply
use this one.

**5.)** The ``closure`` is completely network-dependent and thus has to
remain an abstract method here.

**6.)** The ``prepare_batch`` function also depends on the combination
of network, inputs and loss functions to use, but we can at least give a
prototype of such an function, which handles the devices correctly and
converts everything to ``float``

Actual Implementation
~~~~~~~~~~~~~~~~~~~~~

Now, let's start with the actual implementation and do one function by
another and keep the things in mind, we just discovered.

Class Signature and ``__init__``-Method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To subclass both networks, we cannot use the simple ``super().__init__``
approach, because we have to init both parent classes, so we do

.. code:: python


        class AbstractTorchScriptNetwork(AbstractNetwork, torch.jit.ScriptModule):

            @abc.abstractmethod
            def __init__(self, optimize=True, **kwargs):
                """

                Parameters
                ----------
                optimize : bool
                    whether to optimize the network graph or not; default: True
                **kwargs :
                    additional keyword arguments (passed to :class:`AbstractNetwork`)
                """
                torch.jit.ScriptModule.__init__(self, optimize=optimize)
                AbstractNetwork.__init__(self, **kwargs)
                

instead. This ensures all parent classes to be initialized correctly.

``__call__``-Method
^^^^^^^^^^^^^^^^^^^

As mentioned above, the ``__call__`` method is very easy to implement,
because we can simply use the implementation of our ``TorchScript`` base
class like this:

.. code:: python


        def __call__(self, *args, **kwargs):
            """
            Calls Forward method

            Parameters
            ----------
            *args :
                positional arguments (passed to `forward`)
            **kwargs :
                keyword arguments (passed to `forward`)

            Returns
            -------
            Any
                result: module results of arbitrary type and number

            """
            return torch.jit.ScriptModule.__call__(self, *args, **kwargs)
            

This also ensures, that we can pass an arbitrary number or positional
and keyword arguments of arbitrary types to it (which are all passed to
the ``forward``-function). The advantage over directly calling the
``forward`` method here, is that the ``ScriptModule.__call__`` already
does the handling of
`forward-pre-hooks <https://pytorch.org/docs/stable/nn.html#torch.nn.Module.register_forward_pre_hook>`__,
`forward-hooks <https://pytorch.org/docs/stable/nn.html#torch.nn.Module.register_forward_hook>`__
and
`backward-hooks <https://pytorch.org/docs/stable/nn.html#torch.nn.Module.register_backward_hook>`__.

``closure``-Method
^^^^^^^^^^^^^^^^^^

Since this method is highly model-dependant, we just don't implement it,
which forces the user to implement it (since it is marked as an
``abstractmethod`` in ``AbstractExperiment``).

``prepare_batch``-Method
^^^^^^^^^^^^^^^^^^^^^^^^

The above mentioned prototype of pushing everything to the correct
device and convert it to float looks like this:

.. code:: python


        @staticmethod
        def prepare_batch(batch: dict, input_device, output_device):
            """
            Helper Function to prepare Network Inputs and Labels (convert them to
            correct type and shape and push them to correct devices)

            Parameters
            ----------
            batch : dict
                dictionary containing all the data
            input_device : torch.device
                device for network inputs
            output_device : torch.device
                device for network outputs

            Returns
            -------
            dict
                dictionary containing data in correct type and shape and on correct
                device

            """
            return_dict = {"data": torch.from_numpy(batch.pop("data")).to(
                input_device).to(torch.float)}

            for key, vals in batch.items():
                return_dict[key] = torch.from_numpy(vals).to(output_device).to(
                    torch.float)

            return return_dict

Since we don't want to use any of the model's attributes here (and for
conformity with the ``AbstractNetwork`` class), this method is defined
as ``staticmethod``, meaning it is class-bound, not instance-bound. The
``closure`` method has to be a ``staticmethod`` too.

``forward``-Method
^^^^^^^^^^^^^^^^^^

The only thing left now, is the ``forward`` method, which is internally
called by ``ScriptModule.__call__``. The bad news is: We currently can't
implement it. Subclassing a ``ScriptModule`` to overwrite a function
decorated with ``torch.jit.script_method`` is not (yet) supported, but
will be soon, once `this
PR <https://github.com/pytorch/pytorch/pull/20503>`__ is merged and
released.

For now: you simply have to implement this method in your own network
despite the missing of an abstract interface-method.

Putting it all together
^^^^^^^^^^^^^^^^^^^^^^^

If we combine all the function implementations to one class, it looks
like this:

.. code:: python


        class AbstractTorchScriptNetwork(AbstractNetwork, torch.jit.ScriptModule):

            """
            Abstract Interface Class for TorchScript Networks. For more information
            have a look at https://pytorch.org/docs/stable/jit.html#torchscript

            Warnings
            --------
            In addition to the here defined API, a forward function must be
            implemented and decorated with ``@torch.jit.script_method``

            """
            @abc.abstractmethod
            def __init__(self, optimize=True, **kwargs):
                """

                Parameters
                ----------
                optimize : bool
                    whether to optimize the network graph or not; default: True
                **kwargs :
                    additional keyword arguments (passed to :class:`AbstractNetwork`)
                """
                torch.jit.ScriptModule.__init__(self, optimize=optimize)
                AbstractNetwork.__init__(self, **kwargs)

            def __call__(self, *args, **kwargs):
                """
                Calls Forward method

                Parameters
                ----------
                *args :
                    positional arguments (passed to `forward`)
                **kwargs :
                    keyword arguments (passed to `forward`)

                Returns
                -------
                Any
                    result: module results of arbitrary type and number

                """
                return torch.jit.ScriptModule.__call__(self, *args, **kwargs)

            @staticmethod
            def prepare_batch(batch: dict, input_device, output_device):
                """
                Helper Function to prepare Network Inputs and Labels (convert them to
                correct type and shape and push them to correct devices)

                Parameters
                ----------
                batch : dict
                    dictionary containing all the data
                input_device : torch.device
                    device for network inputs
                output_device : torch.device
                    device for network outputs

                Returns
                -------
                dict
                    dictionary containing data in correct type and shape and on correct
                    device

                """
                return_dict = {"data": torch.from_numpy(batch.pop("data")).to(
                    input_device).to(torch.float)}

                for key, vals in batch.items():
                    return_dict[key] = torch.from_numpy(vals).to(output_device).to(
                        torch.float)

                return return_dict
            

Saving and loading
------------------

Now that we have the ability to implement ``delira``-suitable
TorchScript models, we want to store them on disk and load them again,
so that we don't have to retrain them every time we want to use them.
These I/O functions are usually located in ``delira.io``.

Saving
~~~~~~

Our saving function utilizes multiple functions: ``torch.jit.save`` to
simply save the model (including it's graph) and the
``save_checkpoint_torch`` function implemented for the ``PyTorch``
backend to store the trainer state, since ``TorchScript`` allows us to
use plain ``PyTorch`` optimizers.

The implementation of the function looks like this:

.. code:: python


        def save_checkpoint_torchscript(file: str, model=None, optimizers={},
                                        epoch=None, **kwargs):
            """
            Save current checkpoint to two different files:
                1.) ``file + "_model.ptj"``: Will include the state of the model
                    (including the graph; this is the opposite to
                    :func:`save_checkpoint`)
                2.) ``file + "_trainer_state.pt"``: Will include the states of all
                    optimizers and the current epoch (if given)

            Parameters
            ----------
            file : str
                filepath the model should be saved to
            model : AbstractPyTorchJITNetwork or None
                the model which should be saved
                if None: empty dict will be saved as state dict
            optimizers : dict
                dictionary containing all optimizers
            epoch : int
                current epoch (will also be pickled)

            """

            # remove file extension if given
            if any([file.endswith(ext) for ext in [".pth", ".pt", ".ptj"]]):
                file = file.rsplit(".", 1)[0]

            if isinstance(model, AbstractPyTorchJITNetwork):
                torch.jit.save(model, file + "_model.ptj")

            if optimizers or epoch is not None:
                save_checkpoint_torch(file + "_trainer_state.pt", None,
                                optimizers=optimizers, epoch=epoch, **kwargs)
                

Loading
~~~~~~~

To load a model, which has been saved to disk by this function we have
to revert each part of it. We do this by using ``torch.jit.load`` for
the model (and the graph) and ``load_checkpoint_torch`` by the
``PyTorch`` backend. The actual implementation is given here:

.. code:: python


        def load_checkpoint_torchscript(file: str, **kwargs):
            """
            Loads a saved checkpoint consisting of 2 files
            (see :func:`save_checkpoint_jit` for details)

            Parameters
            ----------
            file : str
                filepath to a file containing a saved model
            **kwargs:
                Additional keyword arguments (passed to torch.load)
                Especially "map_location" is important to change the device the
                state_dict should be loaded to

            Returns
            -------
            OrderedDict
                checkpoint state_dict

            """
            # remove file extensions
            if any([file.endswith(ext) for ext in [".pth", ".pt", ".ptj"]]):
                file = file.rsplit(".", 1)[0]

            # load model
            if os.path.isfile(file + ".ptj"):
                model_file = file
            elif os.path.isfile(file + "_model.ptj"):
                model_file = file + "_model.ptj"
            else:
                raise ValueError("No Model File found for %s" % file)

            # load trainer state (if possible)
            trainer_file = model_file.replace("_model.ptj", "_trainer_state.pt")
            if os.path.isfile(trainer_file):
                trainer_state = load_checkpoint_torch(trainer_file, **kwargs)

            else:
                trainer_state = {"optimizer": {},
                                 "epoch": None}

            trainer_state.update({"model": torch.jit.load(model_file)})

            return trainer_state
        

A Trainer to train
------------------

Now, that we can define and save/load our models, we want to train them.
Luckily ``delira`` has already implemented a very modular
backend-agnostic trainer (the ``BaseNetworkTrainer``) and build upon
this a ``PyTorchNetworkTrainer``. Since the training process in PyTorch
and TorchScript is nearly the same, we can just extend the
``PyTorchNetworkTrainer``. Usually one would have to extend the
``BaseNetworkTrainer`` to provide some backend specific functions (like
necessary initializations, optimizer setup, seeding etc.). To see how
this is done, you could either have a look at the
``PyTorchNetworkTrainer`` or the ``TfNetworkTrainer`` for tensorflow,
which are both following this principle. Usually the only stuff to
completely change is the loading/saving behavior and the ``_setup``
function, which defines the backend-specific initialization. Some other
functions may have to be extended (by implementing the extension and
calling the parent-classes function).

Things to change:
~~~~~~~~~~~~~~~~~

By Subclassing the ``PyTorchNetworkTrainer`` we have to change the
following things:

-  The trainer's default arguments

-  The behavior for trying to resume a previous training

-  The saving, loading and updating behavior

We will access this one by one:

The Default Arguments
^^^^^^^^^^^^^^^^^^^^^

We want to use ``AbstractTorchScriptNetwork``\ s instead of
``AbstractPyTorchNetwork``\ s here and we have to change the behavior if
passing multiple GPUs, because currently Multi-GPU training is not
supported by ``TorchScript``.

To do this: we implement the functions ``__init__``, apply our changes
and forward these changes to the call of the base-classes ``__init__``
like this (omitted docstrings for the sake of shortness):

.. code:: python


    class TorchScriptNetworkTrainer(PyTorchNetworkTrainer):
            def __init__(self,
                         network: AbstractTorchScriptNetwork,
                         save_path: str,
                         key_mapping,
                         losses=None,
                         optimizer_cls=None,
                         optimizer_params={},
                         train_metrics={},
                         val_metrics={},
                         lr_scheduler_cls=None,
                         lr_scheduler_params={},
                         gpu_ids=[],
                         save_freq=1,
                         optim_fn=create_optims_default,
                         logging_type="tensorboardx",
                         logging_kwargs={},
                         fold=0,
                         callbacks=[],
                         start_epoch=1,
                         metric_keys=None,
                         convert_batch_to_npy_fn=convert_torch_tensor_to_npy,
                         criterions=None,
                         val_freq=1,
                         **kwargs):
                
                if len(gpu_ids) > 1:
                    # only use first GPU due to
                    # https://github.com/pytorch/pytorch/issues/15421
                    gpu_ids = [gpu_ids[0]]
                    logging.warning("Multiple GPUs specified. Torch JIT currently "
                                    "supports only single-GPU training. "
                                    "Switching to use only the first GPU for now...")

                super().__init__(network=network, save_path=save_path,
                                 key_mapping=key_mapping, losses=losses,
                                 optimizer_cls=optimizer_cls,
                                 optimizer_params=optimizer_params,
                                 train_metrics=train_metrics,
                                 val_metrics=val_metrics,
                                 lr_scheduler_cls=lr_scheduler_cls,
                                 lr_scheduler_params=lr_scheduler_params,
                                 gpu_ids=gpu_ids, save_freq=save_freq,
                                 optim_fn=optim_fn, logging_type=logging_type,
                                 logging_kwargs=logging_kwargs, fold=fold,
                                 callbacks=callbacks,
                                 start_epoch=start_epoch, metric_keys=metric_keys,
                                 convert_batch_to_npy_fn=convert_batch_to_npy_fn,
                                 mixed_precision=False, mixed_precision_kwargs={},
                                 criterions=criterions, val_freq=val_freq, **kwargs
                                 )
                

Resuming Training
^^^^^^^^^^^^^^^^^

For resuming the training, we have to completely change the
``try_resume_training`` function and cannot reuse the parent's
implementation of it. Thus, we don't call
``super().try_resume_training`` here, but completely reimplement it from
scratch:

.. code:: python


        def try_resume_training(self):
            """
            Load the latest state of a previous training if possible

            """
            # Load latest epoch file if available
            if os.path.isdir(self.save_path):
                # check all files in directory starting with "checkpoint" and
                # not ending with "_best.pth"
                files = [x for x in os.listdir(self.save_path)
                         if os.path.isfile(os.path.join(self.save_path, x))
                         and x.startswith("checkpoint")
                         and not x.endswith("_best.ptj")
                         ]

                # if list is not empty: load previous state
                if files:

                    latest_epoch = max([
                        int(x.rsplit("_", 1)[-1].rsplit(".", 1)[0])
                        for x in files])

                    latest_state_path = os.path.join(self.save_path,
                                                     "checkpoint_epoch_%d.ptj"
                                                     % latest_epoch)

                    # if pth file does not exist, load pt file instead
                    if not os.path.isfile(latest_state_path):
                        latest_state_path = latest_state_path[:-1]

                    logger.info("Attempting to load state from previous \
                                training from %s" % latest_state_path)
                    try:
                        self.update_state(latest_state_path)
                    except KeyError:
                        logger.warning("Previous State could not be loaded, \
                                        although it exists.Training will be \
                                        restarted")

Saving and Loading
^^^^^^^^^^^^^^^^^^

Now we need to change the saving and loading behavior. As always we try
to reuse as much code as possible to avoid code duplication.

Saving
''''''

To save the current training state, we simply call the
``save_checkpoint_torchscript`` function:

.. code:: python


        def save_state(self, file_name, epoch, **kwargs):
            """
            saves the current state via
            :func:`delira.io.torch.save_checkpoint_jit`

            Parameters
            ----------
            file_name : str
                filename to save the state to
            epoch : int
                current epoch (will be saved for mapping back)
            **kwargs :
                keyword arguments

            """
            if file_name.endswith(".pt") or file_name.endswith(".pth"):
                file_name = file_name.rsplit(".", 1)[0]

            save_checkpoint_torchscript(file_name, self.module, self.optimizers,
                                        **kwargs)
            

Loading
'''''''

To load the training state, we simply return the state loaded by
``load_checkpoint_torchscript``. Since we don't use any arguments of the
trainer itself here, the function is a ``staticmethod``:

.. code:: python


        @staticmethod
        def load_state(file_name, **kwargs):
            """
            Loads the new state from file via
            :func:`delira.io.torch.load_checkpoint:jit`

            Parameters
            ----------
            file_name : str
                the file to load the state from
            **kwargs : keyword arguments

            Returns
            -------
            dict
                new state

            """
            return load_checkpoint_torchscript(file_name, **kwargs)
        

Updating
''''''''

After we loaded the new state, we need to update the trainer's internal
state by this new state.

We do this by directly assigning the model here (since the graph was
stored/loaded too) instead of only updating the state\_dict and calling
the parent-classes method afterwards:

.. code:: python


        def _update_state(self, new_state):
            """
            Update the state from a given new state

            Parameters
            ----------
            new_state : dict
                new state to update internal state from

            Returns
            -------
            :class:`PyTorchNetworkJITTrainer`
                the trainer with a modified state

            """
            if "model" in new_state:
                self.module = new_state.pop("model").to(self.input_device)

            return super()._update_state(new_state)

A Whole Trainer
~~~~~~~~~~~~~~~

After combining all the changes above, we finally get our new trainer
as:

.. code:: python


        class TorchScriptNetworkTrainer(PyTorchNetworkTrainer):
            def __init__(self,
                         network: AbstractTorchScriptNetwork,
                         save_path: str,
                         key_mapping,
                         losses=None,
                         optimizer_cls=None,
                         optimizer_params={},
                         train_metrics={},
                         val_metrics={},
                         lr_scheduler_cls=None,
                         lr_scheduler_params={},
                         gpu_ids=[],
                         save_freq=1,
                         optim_fn=create_optims_default,
                         logging_type="tensorboardx",
                         logging_kwargs={},
                         fold=0,
                         callbacks=[],
                         start_epoch=1,
                         metric_keys=None,
                         convert_batch_to_npy_fn=convert_torch_tensor_to_npy,
                         criterions=None,
                         val_freq=1,
                         **kwargs):
                """

                Parameters
                ----------
                network : :class:`AbstractPyTorchJITNetwork`
                    the network to train
                save_path : str
                    path to save networks to
                key_mapping : dict
                    a dictionary containing the mapping from the ``data_dict`` to
                    the actual model's inputs.
                    E.g. if a model accepts one input named 'x' and the data_dict
                    contains one entry named 'data' this argument would have to
                    be ``{'x': 'data'}``
                losses : dict
                    dictionary containing the training losses
                optimizer_cls : subclass of tf.train.Optimizer
                    optimizer class implementing the optimization algorithm of
                    choice
                optimizer_params : dict
                    keyword arguments passed to optimizer during construction
                train_metrics : dict, optional
                    metrics, which will be evaluated during train phase
                    (should work on framework's tensor types)
                val_metrics : dict, optional
                    metrics, which will be evaluated during test phase
                    (should work on numpy arrays)
                lr_scheduler_cls : Any
                    learning rate schedule class: must implement step() method
                lr_scheduler_params : dict
                    keyword arguments passed to lr scheduler during construction
                gpu_ids : list
                    list containing ids of GPUs to use; if empty: use cpu instead
                    Currently ``torch.jit`` only supports single GPU-Training,
                    thus only the first GPU will be used if multiple GPUs are passed
                save_freq : int
                    integer specifying how often to save the current model's state.
                    State is saved every state_freq epochs
                optim_fn : function
                    creates a dictionary containing all necessary optimizers
                logging_type : str or callable
                    the type of logging. If string: it must be one of
                    ["visdom", "tensorboardx"]
                    If callable: it must be a logging handler class
                logging_kwargs : dict
                    dictionary containing all logging keyword arguments
                fold : int
                    current cross validation fold (0 per default)
                callbacks : list
                    initial callbacks to register
                start_epoch : int
                    epoch to start training at
                metric_keys : dict
                    dict specifying which batch_dict entry to use for which metric as
                    target; default: None, which will result in key "label" for all
                    metrics
                convert_batch_to_npy_fn : type, optional
                    function converting a batch-tensor to numpy, per default this is
                    a function, which detaches the tensor, moves it to cpu and the
                    calls ``.numpy()`` on it
                mixed_precision : bool
                    whether to use mixed precision or not (False per default)
                mixed_precision_kwargs : dict
                    additional keyword arguments for mixed precision
                val_freq : int
                    validation frequency specifying how often to validate the trained
                    model (a value of 1 denotes validating every epoch,
                    a value of 2 denotes validating every second epoch etc.);
                    defaults to 1
                **kwargs :
                    additional keyword arguments

                """

                if len(gpu_ids) > 1:
                    # only use first GPU due to
                    # https://github.com/pytorch/pytorch/issues/15421
                    gpu_ids = [gpu_ids[0]]
                    logging.warning("Multiple GPUs specified. Torch JIT currently "
                                    "supports only single-GPU training. "
                                    "Switching to use only the first GPU for now...")

                super().__init__(network=network, save_path=save_path,
                                 key_mapping=key_mapping, losses=losses,
                                 optimizer_cls=optimizer_cls,
                                 optimizer_params=optimizer_params,
                                 train_metrics=train_metrics,
                                 val_metrics=val_metrics,
                                 lr_scheduler_cls=lr_scheduler_cls,
                                 lr_scheduler_params=lr_scheduler_params,
                                 gpu_ids=gpu_ids, save_freq=save_freq,
                                 optim_fn=optim_fn, logging_type=logging_type,
                                 logging_kwargs=logging_kwargs, fold=fold,
                                 callbacks=callbacks,
                                 start_epoch=start_epoch, metric_keys=metric_keys,
                                 convert_batch_to_npy_fn=convert_batch_to_npy_fn,
                                 mixed_precision=False, mixed_precision_kwargs={},
                                 criterions=criterions, val_freq=val_freq, **kwargs
                                 )

            def try_resume_training(self):
                """
                Load the latest state of a previous training if possible

                """
                # Load latest epoch file if available
                if os.path.isdir(self.save_path):
                    # check all files in directory starting with "checkpoint" and
                    # not ending with "_best.pth"
                    files = [x for x in os.listdir(self.save_path)
                             if os.path.isfile(os.path.join(self.save_path, x))
                             and x.startswith("checkpoint")
                             and not x.endswith("_best.ptj")
                             ]

                    # if list is not empty: load previous state
                    if files:

                        latest_epoch = max([
                            int(x.rsplit("_", 1)[-1].rsplit(".", 1)[0])
                            for x in files])

                        latest_state_path = os.path.join(self.save_path,
                                                         "checkpoint_epoch_%d.ptj"
                                                         % latest_epoch)

                        # if pth file does not exist, load pt file instead
                        if not os.path.isfile(latest_state_path):
                            latest_state_path = latest_state_path[:-1]

                        logger.info("Attempting to load state from previous \
                                    training from %s" % latest_state_path)
                        try:
                            self.update_state(latest_state_path)
                        except KeyError:
                            logger.warning("Previous State could not be loaded, \
                                            although it exists.Training will be \
                                            restarted")

            def save_state(self, file_name, epoch, **kwargs):
                """
                saves the current state via
                :func:`delira.io.torch.save_checkpoint_jit`

                Parameters
                ----------
                file_name : str
                    filename to save the state to
                epoch : int
                    current epoch (will be saved for mapping back)
                **kwargs :
                    keyword arguments

                """
                if file_name.endswith(".pt") or file_name.endswith(".pth"):
                    file_name = file_name.rsplit(".", 1)[0]

                save_checkpoint_torchscript(file_name, self.module, self.optimizers,
                                            **kwargs)

            @staticmethod
            def load_state(file_name, **kwargs):
                """
                Loads the new state from file via
                :func:`delira.io.torch.load_checkpoint:jit`

                Parameters
                ----------
                file_name : str
                    the file to load the state from
                **kwargs : keyword arguments

                Returns
                -------
                dict
                    new state

                """
                return load_checkpoint_torchscript(file_name, **kwargs)

            def _update_state(self, new_state):
                """
                Update the state from a given new state

                Parameters
                ----------
                new_state : dict
                    new state to update internal state from

                Returns
                -------
                :class:`PyTorchNetworkJITTrainer`
                    the trainer with a modified state

                """
                if "model" in new_state:
                    self.module = new_state.pop("model").to(self.input_device)

                return super()._update_state(new_state)
            

Wrapping it all in an Experiment
--------------------------------

To have access to methods like a K-Fold (and the not yet finished)
hyperparameter tuning, we need to wrap the trainer in an Experiment. We
will use the same approach as we did for implementing the trainer:
Extending an already provided class.

This time we extend the ``PyTorchExperiment`` which itself extends the
``BaseExperiment`` by some backend-specific defaults, types and seeds.

Our whole class definition just changes the default arguments of the
``PyTorchExperiment`` and thus, we only have to implenent it's
``__init__``:

.. code:: python


    class TorchScriptExperiment(PyTorchExperiment):
        def __init__(self,
                     params: typing.Union[str, Parameters],
                     model_cls: AbstractTorchScriptNetwork, # not AbstractPyTorchNetwork anymore
                     n_epochs=None,
                     name=None,
                     save_path=None,
                     key_mapping=None,
                     val_score_key=None,
                     optim_builder=create_optims_default_pytorch,
                     checkpoint_freq=1,
                     trainer_cls=TorchScriptNetworkTrainer, # not PyTorchNetworkTrainer anymore
                     **kwargs):
            """

            Parameters
            ----------
            params : :class:`Parameters` or str
                the training parameters, if string is passed,
                it is treated as a path to a pickle file, where the
                parameters are loaded from
            model_cls : Subclass of :class:`AbstractTorchScriptNetwork`
                the class implementing the model to train
            n_epochs : int or None
                the number of epochs to train, if None: can be specified later
                during actual training
            name : str or None
                the Experiment's name
            save_path : str or None
                the path to save the results and checkpoints to.
                if None: Current working directory will be used
            key_mapping : dict
                mapping between data_dict and model inputs (necessary for
                prediction with :class:`Predictor`-API), if no keymapping is
                given, a default key_mapping of {"x": "data"} will be used here
            val_score_key : str or None
                key defining which metric to use for validation (determining
                best model and scheduling lr); if None: No validation-based
                operations will be done (model might still get validated,
                but validation metrics can only be logged and not used further)
            optim_builder : function
                Function returning a dict of backend-specific optimizers.
                defaults to :func:`create_optims_default_pytorch`
            checkpoint_freq : int
                frequency of saving checkpoints (1 denotes saving every epoch,
                2 denotes saving every second epoch etc.); default: 1
            trainer_cls : subclass of :class:`TorchScriptNetworkTrainer`
                the trainer class to use for training the model, defaults to
                :class:`TorchScriptNetworkTrainer`
            **kwargs :
                additional keyword arguments

            """
            super().__init__(params=params, model_cls=model_cls,
                             n_epochs=n_epochs, name=name, save_path=save_path,
                             key_mapping=key_mapping,
                             val_score_key=val_score_key,
                             optim_builder=optim_builder,
                             checkpoint_freq=checkpoint_freq,
                             trainer_cls=trainer_cls,
                             **kwargs)
            

Testing it
----------

Now that we finished the implementation of the backend (which is the
outermost wrapper; Congratulations!), we can just test it. We'll use a
very simple network and test it with dummy data. We also only test the
``run`` and ``test`` functionality of our experiment, since everything
else is just used for setting up the internal state or a composition of
these two methods and already tested: Now, let's just define our
dataset, instantiate it three times (for training, validation and
testing) and wrap each of them into a ``BaseDataManager``:

.. code:: ipython3

    from delira.data_loading import AbstractDataset
    from delira.data_loading import BaseDataManager
    
    
    class DummyDataset(AbstractDataset):
        def __init__(self, length):
            super().__init__(None, None)
            self.length = length
    
        def __getitem__(self, index):
            return {"data": np.random.rand(32),
                    "label": np.random.randint(0, 1, 1)}
    
        def __len__(self):
            return self.length
    
        def get_sample_from_index(self, index):
            return self.__getitem__(index)
        
    dset_train = DummyDataset(500)
    dset_val = DummyDataset(50)
    dset_test = DummyDataset(10)
    
    # training, validation and testing with 
    #a batchsize of 16, 1 loading thread and no transformations.
    dmgr_train = BaseDataManager(dset_train, 16, 1, None)
    dmgr_val = BaseDataManager(dset_val, 16, 1, None)
    dmgr_test = BaseDataManager(dset_test, 16, 1, None)

Now, that we have created three datasets, we need to define our small
dummy network. We do this by subclassing
``delira.models.AbstractTorchScriptNetwork`` (which is the exactly
implementation given above, be we need to use the internal one, because
there are some typechecks against this one).

.. code:: ipython3

    from delira.models import AbstractTorchScriptNetwork
    import torch
    
    
    class DummyNetworkTorchScript(AbstractTorchScriptNetwork):
        __constants__ = ["module"]
    
        def __init__(self):
            super().__init__()
            self.module = self._build_model(32, 1)
    
        @torch.jit.script_method
        def forward(self, x):
            return {"pred": self.module(x)}
    
        @staticmethod
        def prepare_batch(batch_dict, input_device, output_device):
            return {"data": torch.from_numpy(batch_dict["data"]
                                             ).to(input_device,
                                                  torch.float),
                    "label": torch.from_numpy(batch_dict["label"]
                                              ).to(output_device,
                                                   torch.float)}
    
        @staticmethod
        def closure(model: AbstractTorchScriptNetwork, data_dict: dict,
                    optimizers: dict, losses={}, metrics={},
                    fold=0, **kwargs):
            """
            closure method to do a single backpropagation step
    
    
            Parameters
            ----------
            model : 
                trainable model
            data_dict : dict
                dictionary containing the data
            optimizers : dict
                dictionary of optimizers to optimize model's parameters
            losses : dict
                dict holding the losses to calculate errors
                (gradients from different losses will be accumulated)
            metrics : dict
                dict holding the metrics to calculate
            fold : int
                Current Fold in Crossvalidation (default: 0)
            **kwargs:
                additional keyword arguments
    
            Returns
            -------
            dict
                Metric values (with same keys as input dict metrics)
            dict
                Loss values (with same keys as input dict losses)
            list
                Arbitrary number of predictions as torch.Tensor
    
            Raises
            ------
            AssertionError
                if optimizers or losses are empty or the optimizers are not
                specified
    
            """
    
            assert (optimizers and losses) or not optimizers, \
                "Criterion dict cannot be emtpy, if optimizers are passed"
    
            loss_vals = {}
            metric_vals = {}
            total_loss = 0
    
            # choose suitable context manager:
            if optimizers:
                context_man = torch.enable_grad
    
            else:
                context_man = torch.no_grad
    
            with context_man():
    
                inputs = data_dict.pop("data")
                preds = model(inputs)
    
                if data_dict:
    
                    for key, crit_fn in losses.items():
                        _loss_val = crit_fn(preds["pred"], *data_dict.values())
                        loss_vals[key] = _loss_val.item()
                        total_loss += _loss_val
    
                    with torch.no_grad():
                        for key, metric_fn in metrics.items():
                            metric_vals[key] = metric_fn(
                                preds["pred"], *data_dict.values()).item()
    
            if optimizers:
                optimizers['default'].zero_grad()
                # perform loss scaling via apex if half precision is enabled
                with optimizers["default"].scale_loss(total_loss) as scaled_loss:
                    scaled_loss.backward()
                optimizers['default'].step()
    
            else:
    
                # add prefix "val" in validation mode
                eval_loss_vals, eval_metrics_vals = {}, {}
                for key in loss_vals.keys():
                    eval_loss_vals["val_" + str(key)] = loss_vals[key]
    
                for key in metric_vals:
                    eval_metrics_vals["val_" + str(key)] = metric_vals[key]
    
                loss_vals = eval_loss_vals
                metric_vals = eval_metrics_vals
    
            return metric_vals, loss_vals, {k: v.detach()
                                            for k, v in preds.items()}
    
        @staticmethod
        def _build_model(in_channels, n_outputs):
            return torch.nn.Sequential(
                torch.nn.Linear(in_channels, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, n_outputs)
            )

Now, that we defined our model, let's just test, if we really can
forward some tensors through it. We will just use some random
``torch.Tensors`` (created by ``torch.rand``). Since our model accepts
1d inputs of length 32, we need to pass 2d tensors to it (the additional
dimension is the batch-dimension).

.. code:: ipython3

    input_tensor_single = torch.rand(1, 32) # use a single-sample batch (batchsize=1) here
    input_tensor_batched = torch.rand(4, 32) # use a batch with batchsize 4 here
    
    # create model instance
    model = DummyNetworkTorchScript()
    
    outputs = {"single": model(input_tensor_single)["pred"], "batched": model(input_tensor_batched)["pred"]}
    outputs




.. parsed-literal::

    {'single': tensor([[-0.1934]], grad_fn=<DifferentiableGraphBackward>),
     'batched': tensor([[-0.0525],
             [-0.0884],
             [-0.1492],
             [-0.0431]], grad_fn=<DifferentiableGraphBackward>)}



.. code:: ipython3

    from sklearn.metrics import mean_absolute_error
    from delira.training.callbacks import ReduceLROnPlateauCallbackPyTorch
    from delira.training import Parameters
    params = Parameters(fixed_params={
                        "model": {},
                        "training": {
                            "losses": {"CE": torch.nn.BCEWithLogitsLoss()},
                            "optimizer_cls": torch.optim.Adam,
                            "optimizer_params": {"lr": 1e-3},
                            "num_epochs": 2,
                            "val_metrics": {"mae": mean_absolute_error},
                            "lr_sched_cls": ReduceLROnPlateauCallbackPyTorch,
                            "lr_sched_params": {"mode": "min"}
                        }
                    }
              )
    
    from delira.training import TorchScriptExperiment
    
    exp = TorchScriptExperiment(params, DummyNetworkTorchScript,
                                key_mapping={"x": "data"},
                                val_score_key="mae",
                                val_score_mode="min")
    
    trained_model = exp.run(dmgr_train, dmgr_val)
    exp.test(trained_model, dmgr_test, params.nested_get("val_metrics"))

Congratulations. You have implemented your first fully-workable
``delira``-Backend. Wasn't that hard, was it?

Before you start implementing backends for all the other frameworks out
there, let me just give you some advices:

-  You should test everything you implement or extend

-  Make sure, to keep your backend-specification in mind

-  Always follow the API of already existing backends. If this is not
   possible: test this extensively

-  If you extend another backend (like we did here; we extended the
   ``PyTorch``-backend for ``TorchScript``), make sure, that the
   "base-backend" is always installed (best if they can only be
   installed together)

-  If you have questions regarding the implementation, don't hestiate to
   contact us.
