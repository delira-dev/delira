
Delira Introduction
===================

Authors: Justus Schock, Christoph Haarburger

Loading Data
------------

To train your network you first need to load your training data (and
probably also your validation data). This chapter will therefore deal
with ``delira``'s capabilities to load your data (and apply some
augmentation).

The Dataset
~~~~~~~~~~~

There are mainly two ways to load your data: Lazy or non-lazy. Loading
in a lazy way means that you load the data just in time and keep the
used memory to a bare minimum. This has, however, the disadvantage that
your loading function could be a bottleneck since all postponed
operations may have to wait until the needed data samples are loaded. In
a no-lazy way, one would preload all data to RAM before starting any
other operations. This has the advantage that there cannot be a loading
bottleneck during latter operations. This advantage comes at cost of a
higher memory usage and a (possibly) huge latency at the beginning of
each experiment. Both ways to load your data are implemented in
``delira`` and they are named ``BaseLazyDataset``\ and
``BaseCacheDataset``. In the following steps you will only see the
``BaseLazyDataset`` since exchanging them is trivial. All Datasets
(including the ones you might want to create yourself later) must be
derived of ``delira.data_loading.AbstractDataset`` to ensure a minimum
common API.

The dataset's ``__init__`` has the following signature:

.. code:: python

    def __init__(self, data_path, load_fn, img_extensions, gt_extensions,
                     **load_kwargs):

This means, you have to pass the path to the directory containing your
data (``data_path``), a function to load a single sample of your data
(``load_fn``), the file extensions for valid images (``img_extensions``)
and the extensions for valid groundtruth files (``gt_files``). The
defined extensions are used to index all data files in the given
``data_path``. To get a single sample of your dataset after creating it,
you can index it like this: ``dataset[0]``.

The missing argument ``**load_kwargs`` accepts an arbitrary amount of
additional keyword arguments which are directly passed to your loading
function.

An example of how loading your data may look like is given below:

.. code:: python

    from delira.data_loading import BaseLazyDataset, default_load_fn_2d
    dataset_train = BaseLazyDataset("/images/datasets/external/mnist/train",
                                    default_load_fn_2d, img_extensions=[".png"],
                                    gt_extensions=[".txt"], img_shape=(224, 224))

In this case all data lying in ``/images/datasets/external/mnist/train``
is loaded by ``default_load_fn_2d``. The files containing the data must
be PNG-files, while the groundtruth is defined in TXT-files. The
``default_load_fn_2d`` needs the additional argument ``img_shape`` which
is passed as keyword argument via ``**load_kwargs``.

    **Note:** for reproducability we decided to use some wrapped PyTorch
    datasets for this introduction.

Now, let's just initialize our trainset:

.. code:: ipython3

    from delira.data_loading import TorchvisionClassificationDataset
    dataset_train = TorchvisionClassificationDataset("mnist", train=True,
                                                     img_shape=(224, 224))

Getting a single sample of your dataset with dataset\_train[0] will
produce:

.. code:: ipython3

    dataset_train[0]

which means, that our data is stored in a dictionary containing the keys
``data`` and ``label``, each of them holding the corresponding numpy
arrays. The dataloading works on ``numpy`` purely and is thus backend
agnostic. It does not matter in which format or with which library you
load/preprocess your data, but at the end it must be converted to numpy
arrays For validation purposes another dataset could be created with the
test data like this:

.. code:: ipython3

    dataset_val = TorchvisionClassificationDataset("mnist", train=False,
                                                   img_shape=(224, 224))

The Dataloader
~~~~~~~~~~~~~~

The Dataloader wraps your dataset to privode the ability to load whole
batches with an abstract interface. To create a dataloader, one would
have to pass the following arguments to it's ``__init__``: the
previously created ``dataset``.Additionally, it is possible to pass the
``batch_size`` defining the number of samples per batch, the total
number of batches (``num_batches``), which will be the number of samples
in your dataset devided by the batchsize per default, a random
``seed``\ for always getting the same behaviour of random number
generators and a ```sampler`` <>`__ defining your sampling strategy.
This would create a dataloader for your ``dataset_train``:

.. code:: ipython3

    from delira.data_loading import BaseDataLoader
    
    batch_size = 32
    
    loader_train = BaseDataLoader(dataset_train, batch_size)

Since the batch\_size has been set to 32, the loader will load 32
samples as one batch.

Even though it would be possible to train your network with an instance
of ``BaseDataLoader``, ``malira`` also offers a different approach that
covers multithreaded data loading and augmentation:

The Datamanager
~~~~~~~~~~~~~~~

The data manager is implemented as
``delira.data_loading.BaseDataManager`` and wraps a ``DataLoader``. It
also encapsulates augmentations. Having a view on the
``BaseDataManager``'s signature, it becomes obvious that it accepts the
same arguments as the ```DataLoader`` <#The-Dataloader>`__. You can
either pass a ``dataset`` or a combination of path, dataset class and
load function. Additionally, you can pass a custom dataloder class if
necessary and a sampler class to choose a sampling algorithm.

The parameter ``transforms`` accepts augmentation transformations as
implemented in ``batchgenerators``. Augmentation is applied on the fly
using ``n_process_augmentation`` threads.

All in all the DataManager is the recommended way to generate batches
from your dataset.

The following example shows how to create a data manager instance:

.. code:: ipython3

    from delira.data_loading import BaseDataManager
    from batchgenerators.transforms.abstract_transforms import Compose
    from batchgenerators.transforms.spatial_transforms import MirrorTransform
    from batchgenerators.transforms.sample_normalization_transforms import MeanStdNormalizationTransform
    
    batchsize = 64
    transforms = Compose([MeanStdNormalizationTransform(mean=1*[0], std=1*[1])])
    
    data_manager_train = BaseDataManager(dataset_train,  # dataset to use
                                        batchsize,  # batchsize
                                        n_process_augmentation=1,  # number of augmentation processes
                                        transforms=transforms)  # augmentation transforms


The approach to initialize a DataManager from a datapath takes more
arguments since, in opposite to initializaton from dataset, it needs all
the arguments which are necessary to internally create a dataset.

Since we want to validate our model we have to create a second manager
containing our ``dataset_val``:

.. code:: ipython3

    data_manager_val = BaseDataManager(dataset_val, 
                                        batchsize, 
                                        n_process_augmentation=1, 
                                        transforms=transforms)

That's it - we just finished loading our data!

Iterating over a DataManager is possible in simple loops:

.. code:: ipython3

    from tqdm.auto import tqdm # utility for progress bars
    
    # create actual batch generator from DataManager
    batchgen = data_manager_val.get_batchgen()
    
    for data in tqdm(batchgen):
        pass # here you can access the data of the current batch

Sampler
~~~~~~~

In previous section samplers have been already mentioned but not yet
explained. A sampler implements an algorithm how a batch should be
assembled from single samples in a dataset. ``delira`` provides the
following sampler classes in it's subpackage
``delira.data_loading.sampler``:

-  ``AbstractSampler``
-  ``SequentialSampler``
-  ``PrevalenceSequentialSampler``
-  ``RandomSampler``
-  ``PrevalenceRandomSampler``
-  ``WeightedRandomSampler``
-  ``LambdaSampler``

The ``AbstractSampler`` implements no sampling algorithm but defines a
sampling API and thus all custom samplers must inherit from this class.
The ``Sequential`` sampler builds batches by just iterating over the
samples' indices in a sequential way. Following this, the
``RandomSampler`` builds batches by randomly drawing the samples'
indices with replacement. If the class each sample belongs to is known
for each sample at the beginning, the ``PrevalenceSequentialSampler``
and the ``PrevalenceRandomSampler`` perform a per-class sequential or
random sampling and building each batch with the exactly same number of
samples from each class. The ``WeightedRandomSampler``\ accepts custom
weights to give specific samples a higher probability during random
sampling than others.

The ``LambdaSampler`` is a wrapper for a custom sampling function, which
can be passed to the wrapper during it's initialization, to ensure API
conformity.

It can be passed to the DataLoader or DataManager as class (argument
``sampler_cls``) or as instance (argument ``sampler``).

Models
------

Since the purpose of this framework is to use machine learning
algorithms, there has to be a way to define them. Defining models is
straight forward. ``delira`` provides a class
``delira.models.AbstractNetwork``. *All models must inherit from this
class*.

To inherit this class four functions must be implemented in the
subclass:

-  ``__init__``
-  ``closure``
-  ``prepare_batch``
-  ``__call__``

``__init__``
~~~~~~~~~~~~

The ``__init__``\ function is a classes constructor. In our case it
builds the entire model (maybe using some helper functions). If writing
your own custom model, you have to override this method.

    **Note:** If you want the best experience for saving your model and
    completely recreating it during the loading process you need to take
    care of a few things: \* if using ``torchvision.models`` to build
    your model, always import it with
    ``from torchvision import models as t_models`` \* register all
    arguments in your custom ``__init__`` in the abstract class. A
    init\_prototype could look like this:

.. code:: python

    def __init__(self, in_channels: int, n_outputs: int, **kwargs):
        """

        Parameters
        ----------
        in_channels: int
            number of input_channels
        n_outputs: int
            number of outputs (usually same as number of classes)
        """
        # register params by passing them as kwargs to parent class __init__
        # only params registered like this will be saved!
        super().__init__(in_channels=in_channels,
                         n_outputs=n_outputs,
                         **kwargs)

``closure``
~~~~~~~~~~~

The ``closure``\ function defines one batch iteration to train the
network. This function is needed for the framework to provide a generic
trainer function which works with all kind of networks and loss
functions.

The closure function must implement all steps from forwarding, over loss
calculation, metric calculation, logging (for which
``delira.logging_handlers`` provides some extensions for pythons logging
module), and the actual backpropagation.

It is called with an empty optimizer-dict to evaluate and should thus
work with optional optimizers.

``prepare_batch``
~~~~~~~~~~~~~~~~~

The ``prepare_batch``\ function defines the transformation from loaded
data to match the networks input and output shape and pushes everything
to the right device.

Abstract Networks for specific Backends
---------------------------------------

PyTorch
~~~~~~~

At the time of writing, PyTorch is the only backend which is supported,
but other backends are planned. In PyTorch every network should be
implemented as a subclass of ``torch.nn.Module``, which also provides a
``__call__`` method.

This results in sloghtly different requirements for PyTorch networks:
instead of implementing a ``__call__`` method, we simply call the
``torch.nn.Module.__call__`` and therefore have to implement the
``forward`` method, which defines the module's behaviour and is
internally called by ``torch.nn.Module.__call__`` (among other stuff).
To give a default behaviour suiting most cases and not have to care
about internals, ``delira`` provides the ``AbstractPyTorchNetwork``
which is a more specific case of the ``AbstractNetwork`` for PyTorch
modules.

``forward``
^^^^^^^^^^^

The ``forward`` function defines what has to be done to forward your
input through your network. Assuming your network has three
convolutional layers stored in ``self.conv1``, ``self.conv2`` and
``self.conv3`` and a ReLU stored in ``self.relu``, a simple ``forward``
function could look like this:

.. code:: python

    def forward(self, input_batch: torch.Tensor):
        out_1 = self.relu(self.conv1(input_batch))
        out_2 = self.relu(self.conv2(out_1))
        out_3 = self.conv3(out2)
        
        return out_3

``prepare_batch``
^^^^^^^^^^^^^^^^^

The default ``prepare_batch`` function for PyTorch networks looks like
this:

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
                input_device)}

            for key, vals in batch.items():
                return_dict[key] = torch.from_numpy(vals).to(output_device)

            return return_dict

and can be customized by subclassing the ``AbstractPyTorchNetwork``.

``closure example``
^^^^^^^^^^^^^^^^^^^

A simple closure function for a PyTorch module could look like this:

.. code:: python

        @staticmethod
        def closure(model: AbstractPyTorchNetwork, data_dict: dict,
                    optimizers: dict, criterions={}, metrics={},
                    fold=0, **kwargs):
            """
            closure method to do a single backpropagation step

            Parameters
            ----------
            model : :class:`ClassificationNetworkBasePyTorch`
                trainable model
            data_dict : dict
                dictionary containing the data
            optimizers : dict
                dictionary of optimizers to optimize model's parameters
            criterions : dict
                dict holding the criterions to calculate errors
                (gradients from different criterions will be accumulated)
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
                Loss values (with same keys as input dict criterions)
            list
                Arbitrary number of predictions as torch.Tensor

            Raises
            ------
            AssertionError
                if optimizers or criterions are empty or the optimizers are not
                specified

            """

            assert (optimizers and criterions) or not optimizers, \
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

                    for key, crit_fn in criterions.items():
                        _loss_val = crit_fn(preds, *data_dict.values())
                        loss_vals[key] = _loss_val.detach()
                        total_loss += _loss_val

                    with torch.no_grad():
                        for key, metric_fn in metrics.items():
                            metric_vals[key] = metric_fn(
                                preds, *data_dict.values())

            if optimizers:
                optimizers['default'].zero_grad()
                total_loss.backward()
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

            for key, val in {**metric_vals, **loss_vals}.items():
                logging.info({"value": {"value": val.item(), "name": key,
                                        "env_appendix": "_%02d" % fold
                                        }})

            logging.info({'image_grid': {"images": inputs, "name": "input_images",
                                         "env_appendix": "_%02d" % fold}})

            return metric_vals, loss_vals, [preds]

    **Note:** This closure is taken from the
    ``delira.models.classification.ClassificationNetworkBasePyTorch``

Other examples
~~~~~~~~~~~~~~

In ``delira.models`` you can find exemplaric implementations of
generative adversarial networks, classification and regression
approaches or segmentation networks.

Training
--------

Parameters
~~~~~~~~~~

Training-parameters (often called hyperparameters) can be defined in the
``delira.training.Parameters`` class.

The class accepts the parameters ``batch_size`` and ``num_epochs`` to
define the batchsize and the number of epochs to train, the parameters
``optimizer_cls`` and ``optimizer_params`` to create an optimizer or
training, the parameter ``criterions`` to specify the training
criterions (whose gradients will be accumulated by defaut), the
parameters ``lr_sched_cls`` and ``lr_sched_params`` to define the
learning rate scheduling and the parameter ``metrics`` to specify
evaluation metrics.

Additionally, it is possible to pass an aritrary number of keyword
arguments to the class

It is good practice to create a ``Parameters`` object at the beginning
and then use it for creating other objects which are needed for
training, since you can use the classes attributes and changes in
hyperparameters only have to be done once:

.. code:: ipython3

    import torch
    from delira.training import Parameters
    from delira.data_loading import RandomSampler, SequentialSampler
    
    params = Parameters(fixed_params={
        "model": {},
        "training": {
            "batch_size": 64, # batchsize to use
            "num_epochs": 2, # number of epochs to train
            "optimizer_cls": torch.optim.Adam, # optimization algorithm to use
            "optimizer_params": {'lr': 1e-3}, # initialization parameters for this algorithm
            "criterions": {"CE": torch.nn.CrossEntropyLoss()}, # the loss function
            "lr_sched_cls": None,  # the learning rate scheduling algorithm to use
            "lr_sched_params": {}, # the corresponding initialization parameters
            "metrics": {} # and some evaluation metrics
        }
    }) 
    
    # recreating the data managers with the batchsize of the params object
    manager_train = BaseDataManager(dataset_train, params.nested_get("batch_size"), 1,
                                    transforms=None, sampler_cls=RandomSampler,
                                    n_process_loading=4)
    manager_val = BaseDataManager(dataset_val, params.nested_get("batch_size"), 3,
                                  transforms=None, sampler_cls=SequentialSampler,
                                  n_process_loading=4)


Trainer
~~~~~~~

The ``delira.training.NetworkTrainer`` class provides functions to train
a single network by passing attributes from your parameter object, a
``save_freq`` to specify how often your model should be saved
(``save_freq=1`` indicates every epoch, ``save_freq=2`` every second
epoch etc.) and ``gpu_ids``. If you don't pass any ids at all, your
network will be trained on CPU (and probably take a lot of time). If you
specify 1 id, the network will be trained on the GPU with the
corresponding index and if you pass multiple ``gpu_ids`` your network
will be trained on multiple GPUs in parallel.

    **Note:** The GPU indices are refering to the devices listed in
    ``CUDA_VISIBLE_DEVICES``. E.g if ``CUDA_VISIBLE_DEVICES`` lists GPUs
    3, 4, 5 then gpu\_id 0 will be the index for GPU 3 etc.

    **Note:** training on multiple GPUs is not recommended for easy and
    small networks, since for these networks the synchronization
    overhead is far greater than the parallelization benefit.

Training your network might look like this:

.. code:: ipython3

    from delira.training import PyTorchNetworkTrainer
    from delira.models.classification import ClassificationNetworkBasePyTorch
    
    # path where checkpoints should be saved
    save_path = "./results/checkpoints"
    
    model = ClassificationNetworkBasePyTorch(in_channels=1, n_outputs=10)
    
    trainer = PyTorchNetworkTrainer(network=model,
                                    save_path=save_path,
                                    criterions=params.nested_get("criterions"),
                                    optimizer_cls=params.nested_get("optimizer_cls"),
                                    optimizer_params=params.nested_get("optimizer_params"),
                                    metrics=params.nested_get("metrics"),
                                    lr_scheduler_cls=params.nested_get("lr_sched_cls"),
                                    lr_scheduler_params=params.nested_get("lr_sched_params"),
                                    gpu_ids=[0]
                            )
    
    #trainer.train(params.nested_get("num_epochs"), manager_train, manager_val)


Experiment
~~~~~~~~~~

The ``delira.training.AbstractExperiment`` class needs an experiment
name, a path to save it's results to, a parameter object, a model class
and the keyword arguments to create an instance of this class. It
provides methods to perform a single training and also a method for
running a kfold-cross validation. In order to create it, you must choose
the ``PyTorchExperiment``, which is basically just a subclass of the
``AbstractExperiment`` to provide a general setup for PyTorch modules.
Running an experiment could look like this:

.. code:: ipython3

    from delira.training import PyTorchExperiment
    from delira.training.train_utils import create_optims_default_pytorch
    
    # Add model parameters to Parameter class
    params.fixed.model = {"in_channels": 1, "n_outputs": 10}
    
    experiment = PyTorchExperiment(params=params, 
                                   model_cls=ClassificationNetworkBasePyTorch,
                                   name="TestExperiment", 
                                   save_path="./results",
                                   optim_builder=create_optims_default_pytorch,
                                   gpu_ids=[0])
    
    experiment.run(manager_train, manager_val)

An ``Experiment`` is the most abstract (and recommended) way to define,
train and validate your network.

Logging
-------

Previous class and function definitions used pythons's ``logging``
library. As extensions for this library ``delira`` provides a package
(``delira.logging``) containing handlers to realize different logging
methods.

To use these handlers simply add them to your logger like this:

.. code:: python

    logger.addHandler(logging.StreamHandler())

Nowadays, delira mainly relies on
`trixi <https://github.com/MIC-DKFZ/trixi/>`__ for logging and provides
only a ``MultiStreamHandler`` and a ``TrixiHandler``, which is a binding
to ``trixi``'s loggers and integrates them into the python ``logging``
module

``MultiStreamHandler``
~~~~~~~~~~~~~~~~~~~~~~

The ``MultiStreamHandler`` accepts an arbitrary number of streams during
initialization and writes the message to all of it's streams during
logging.

Logging with ``Visdom`` - The ``trixi`` Loggers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

```Visdom`` <https://github.com/facebookresearch/visdom>`__ is a tool
designed to visualize your logs. To use this tool you need to open a
port on the machine you want to train on via
``visdom -port YOUR_PORTNUMBER`` Afterwards just add the handler of your
choice to the logger. For more detailed information and customization
have a look at `this <https://github.com/facebookresearch/visdom>`__
website.

Logging the scalar tensors containing ``1``, ``2``, ``3``, ``4`` (at the
beginning; will increase to show epochwise logging) with the
corresponding keys ``"one"``, ``"two"``, ``"three"``, ``"four"`` and two
random images with the keys ``"prediction"`` and ``"groundtruth"`` would
look like this:

.. code:: ipython3

    NUM_ITERS = 4
    
    # import logging handler and logging module
    from delira.logging import TrixiHandler
    from trixi.logger import PytorchVisdomLogger
    import logging
    
    # configure logging module (and root logger)
    logger_kwargs = {
        'name': 'test_env', # name of loggin environment
        'port': 9999 # visdom port to connect to
    }
    logger_cls = PytorchVisdomLogger
    
    # configure logging module (and root logger)
    logging.basicConfig(level=logging.INFO,
                        handlers=[TrixiHandler(logger_cls, **logger_kwargs)])
    # derive logger from root logger
    # (don't do `logger = logging.Logger("...")` since this will create a new
    # logger which is unrelated to the root logger
    logger = logging.getLogger("Test Logger")
    
    # create dict containing the scalar numbers as torch.Tensor
    scalars = {"one": torch.Tensor([1]),
               "two": torch.Tensor([2]),
               "three": torch.Tensor([3]),
               "four": torch.Tensor([4])}
    
    # create dict containing the images as torch.Tensor
    # pytorch awaits tensor dimensionality of 
    # batchsize x image channels x height x width
    images = {"prediction": torch.rand(1, 3, 224, 224),
              "groundtruth": torch.rand(1, 3, 224, 224)}
    
    # Simulate 4 Epochs
    for i in range(4*NUM_ITERS): 
        logger.info({"image_grid": {"images": images["prediction"], "name": "predictions"}})
        
        for key, val_tensor in scalars.items():
            logger.info({"value": {"value": val_tensor.item(), "name": key}})
            scalars[key] += 1

    **Note:** The following section is deprecated and is only contained
    for legacy reasons. It is absolutely not recommended to use this
    code ### ``ImgSaveHandler`` The ``ImgSaveHandler`` saves the images
    to a specified directory. The logging message must either include an
    image or a dictionary containing a key 'images' which should be
    associated with a list or dict of images.

Types of VisdomHandlers
^^^^^^^^^^^^^^^^^^^^^^^

The abilities of a handler is simply derivable by it's name: A
``VisdomImageHandler`` is the pure visdom logger, whereas the
``VisdomImageSaveHandler`` combines the abilities of a
``VisdomImageHandler``\ and a ``ImgSaveHandler``. Together with a
``StreamHandler`` (in-built handler) you get the
``VisdomImageStreamHandler`` and if you also want to add the option to
save images to disk, you should use the ``VisdomImageSaveStreamHandler``

The provided handlers are:

-  ``ImgSaveHandler``
-  ``MultistreamHandler``
-  ``VisdomImageHandler``
-  ``VisdomImageSaveHandler``
-  ``VisdomImageSaveStreamHandler``
-  ``VisdomStreamHandler``

More Examples
-------------

More Examples can be found in \* `the classification
example <classification_pytorch.ipynb,>`__ \* `the 2d segmentation
example <segmentation_2d_pytorch.ipynb,>`__ \* `the 3d segmentation
example <segmentation_3d_pytorch.ipynb,>`__ \* `the generative
adversarial example <gan_pytorch.ipynb,>`__
