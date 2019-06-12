
Generative Adversarial Nets with Delira - A very short introduction
===================================================================

*Author: Justus Schock*

*Date: 04.12.2018*

This Example shows how to set up a basic GAN PyTorch experiment and
Visdom Logging Environment.

HyperParameters
---------------

Let's first setup the essential hyperparameters. We will use
``delira``'s ``Parameters``-class for this:

.. code:: ipython3

    logger = None
    import torch
    from delira.training import Parameters
    params = Parameters(fixed_params={
        "model": {
            "n_channels": 1, 
            "noise_length": 10
        },
        "training": {
            "batch_size": 64, # batchsize to use
            "num_epochs": 10, # number of epochs to train
            "optimizer_cls": torch.optim.Adam, # optimization algorithm to use
            "optimizer_params": {'lr': 1e-3}, # initialization parameters for this algorithm
            "losses": {"L1": torch.nn.L1Loss()}, # the loss function
            "lr_sched_cls": None,  # the learning rate scheduling algorithm to use
            "lr_sched_params": {}, # the corresponding initialization parameters
            "metrics": {} # and some evaluation metrics
        }
    }) 

Since we specified ``torch.nn.L1Loss`` as criterion and
``torch.nn.MSELoss`` as metric, they will be both calculated for each
batch, but only the criterion will be used for backpropagation. Since we
have a simple generative task, this should be sufficient. We will train
our network with a batchsize of 64 by using ``Adam`` as optimizer of
choice.

Logging and Visualization
-------------------------

To get a visualization of our results, we should monitor them somehow.
For logging we will use ``Visdom``. To start a visdom server you need to
execute the following command inside an environment which has visdom
installed:

.. code:: shell

    visdom -port=9999

This will start a visdom server on port 9999 of your machine and now we
can start to configure our logging environment. To view your results you
can open http://localhost:9999 in your browser.

.. code:: ipython3

    from trixi.logger import PytorchVisdomLogger
    from delira.logging import TrixiHandler
    import logging
    
    logger_kwargs = {
        'name': 'GANExampleLogger', # name of our logging environment
        'port': 9999 # port on which our visdom server is alive
    }
    
    logger_cls = PytorchVisdomLogger
    
    # configure logging module (and root logger)
    logging.basicConfig(level=logging.INFO,
                        handlers=[TrixiHandler(logger_cls, **logger_kwargs)])
    
    
    # derive logger from root logger
    # (don't do `logger = logging.Logger("...")` since this will create a new
    # logger which is unrelated to the root logger
    logger = logging.getLogger("Test Logger")
    

Since a single visdom server can run multiple environments, we need to
specify a (unique) name for our environment and need to tell the logger,
on which port it can find the visdom server.

Data Preparation
----------------

Loading
~~~~~~~

Next we will create a small train and validation set (based on
``torchvision`` MNIST):

.. code:: ipython3

    from delira.data_loading import TorchvisionClassificationDataset
    
    dataset_train = TorchvisionClassificationDataset("mnist", # which dataset to use
                                                     train=True, # use trainset
                                                     img_shape=(224, 224) # resample to 224 x 224 pixels
                                                    )
    dataset_val = TorchvisionClassificationDataset("mnist", 
                                                   train=False,
                                                   img_shape=(224, 224)
                                                  )

Augmentation
~~~~~~~~~~~~

For Data-Augmentation we will apply a few transformations:

.. code:: ipython3

    from batchgenerators.transforms import RandomCropTransform, \
                                            ContrastAugmentationTransform, Compose
    from batchgenerators.transforms.spatial_transforms import ResizeTransform
    from batchgenerators.transforms.sample_normalization_transforms import MeanStdNormalizationTransform
    
    transforms = Compose([
        RandomCropTransform(200), # Perform Random Crops of Size 200 x 200 pixels
        ResizeTransform(224), # Resample these crops back to 224 x 224 pixels
        ContrastAugmentationTransform(), # randomly adjust contrast
        MeanStdNormalizationTransform(mean=[0.5], std=[0.5])]) 
    
    

With these transformations we can now wrap our datasets into
datamanagers:

.. code:: ipython3

    from delira.data_loading import BaseDataManager, SequentialSampler, RandomSampler
    
    manager_train = BaseDataManager(dataset_train, params.nested_get("batch_size"),
                                    transforms=transforms,
                                    sampler_cls=RandomSampler,
                                    n_process_augmentation=4)
    
    manager_val = BaseDataManager(dataset_val, params.nested_get("batch_size"),
                                  transforms=transforms,
                                  sampler_cls=SequentialSampler,
                                  n_process_augmentation=4)
    

Training
--------

After we have done that, we can finally specify our experiment and run
it. We will therfore use the already implemented
``GenerativeAdversarialNetworkBasePyTorch`` which is basically a vanilla
DCGAN:

.. code:: ipython3

    import warnings
    warnings.simplefilter("ignore", UserWarning) # ignore UserWarnings raised by dependency code
    warnings.simplefilter("ignore", FutureWarning) # ignore FutureWarnings raised by dependency code
    
    
    from delira.training import PyTorchExperiment
    from delira.training.train_utils import create_optims_gan_default_pytorch
    from delira.models.gan import GenerativeAdversarialNetworkBasePyTorch
    
    if logger is not None:
        logger.info("Init Experiment")
    experiment = PyTorchExperiment(params, GenerativeAdversarialNetworkBasePyTorch,
                                   name="GANExample",
                                   save_path="./tmp/delira_Experiments",
                                   optim_builder=create_optims_gan_default_pytorch,
                                   gpu_ids=[0])
    experiment.save()
    
    model = experiment.run(manager_train, manager_val)

Congratulations, you have now trained your first Generative Adversarial
Model using ``delira``.

See Also
--------

For a more detailed explanation have a look at \* `the introduction
tutorial <tutorial_delira.ipynb,>`__ \* `the 2d segmentation
example <segmentation_2d_pytorch.ipynb,>`__ \* `the 3d segmentation
example <segmentation_3d_pytorch.ipynb,>`__ \* `the classification
example <classification_pytorch.ipynb,>`__
