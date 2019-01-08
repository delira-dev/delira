
Segmentation in 3D using U-Nets with Delira - A very short introduction
=======================================================================

*Author: Justus Schock, Alexander Moriz*

*Date: 17.12.2018*

This Example shows how use the U-Net implementation in Delira with
PyTorch.

Let's first setup the essential hyperparameters. We will use
``delira``'s ``Parameters``-class for this:

.. code:: ipython3

    import torch
    from delira.training import Parameters
    params = Parameters(fixed_params={
        "model": {
            "in_channels": 1, 
            "num_classes": 4
        },
        "training": {
            "batch_size": 64, # batchsize to use
            "num_epochs": 10, # number of epochs to train
            "optimizer_cls": torch.optim.Adam, # optimization algorithm to use
            "optimizer_params": {'lr': 1e-3}, # initialization parameters for this algorithm
            "criterions": {"CE": torch.nn.CrossEntropyLoss()}, # the loss function
            "lr_sched_cls": None,  # the learning rate scheduling algorithm to use
            "lr_sched_params": {}, # the corresponding initialization parameters
            "metrics": {} # and some evaluation metrics
        }
    }) 

Since we did not specify any metric, only the ``CrossEntropyLoss`` will
be calculated for each batch. Since we have a classification task, this
should be sufficient. We will train our network with a batchsize of 64
by using ``Adam`` as optimizer of choice.

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
        'name': 'ClassificationExampleLogger', # name of our logging environment
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

Data Praparation
----------------

Loading
~~~~~~~

Next we will create a small train and validation set (in this case they
will be the same to show the overfitting capability of the UNet).

Our data is a brain MR-image thankfully provided by the
`FSL <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki>`__ in their
`introduction <http://www.fmrib.ox.ac.uk/primers/intro_primer/ExBox3/IntroBox3.html>`__.

We first download the data and extract the T1 image and the
corresponding segmentation:

.. code:: ipython3

    from io import BytesIO
    from zipfile import ZipFile
    from urllib.request import urlopen
    
    resp = urlopen("http://www.fmrib.ox.ac.uk/primers/intro_primer/ExBox3/ExBox3.zip")
    zipfile = ZipFile(BytesIO(resp.read()))
    #zipfile_list = zipfile.namelist()
    #print(zipfile_list)
    img_file = zipfile.extract("ExBox3/T1_brain.nii.gz")
    mask_file = zipfile.extract("ExBox3/T1_brain_seg.nii.gz")

Now, we load the image and the mask (they are both 3D), convert them to
a 32-bit floating point numpy array and ensure, they have the same shape
(i.e. that for each voxel in the image, there is a voxel in the mask):

.. code:: ipython3

    import SimpleITK as sitk
    import numpy as np
    
    # load image and mask
    img = sitk.GetArrayFromImage(sitk.ReadImage(img_file))
    img = img.astype(np.float32)
    mask = mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_file))
    mask = mask.astype(np.float32)
    
    assert mask.shape == img.shape
    print(img.shape)

By querying the unique values in the mask, we get the following:

.. code:: ipython3

    np.unique(mask)

This means, there are 4 classes (background and 3 types of tissue) in
our sample.

To load the data, we have to use a ``Dataset``. The following defines a
very simple dataset, accepting an image slice, a mask slice and the
number of samples. It always returns the same sample until
``num_samples`` samples have been returned.

.. code:: ipython3

    from delira.data_loading import AbstractDataset
    
    class CustomDataset(AbstractDataset):
        def __init__(self, img, mask, num_samples=1000):
            super().__init__(None, None, None, None)
            self.data = {"data": img.reshape(1, *img.shape), "label": mask.reshape(1, *mask.shape)}
            self.num_samples = num_samples
            
        def __getitem__(self, index):
            return self.data
        
        def __len__(self):
            return self.num_samples

Now, we can finally instantiate our datasets:

.. code:: ipython3

    dataset_train = CustomDataset(img, mask, num_samples=10000)
    dataset_val = CustomDataset(img, mask, num_samples=1)

Augmentation
~~~~~~~~~~~~

For Data-Augmentation we will apply a few transformations:

.. code:: ipython3

    from batchgenerators.transforms import RandomCropTransform, \
                                            ContrastAugmentationTransform, Compose
    from batchgenerators.transforms.spatial_transforms import ResizeTransform
    from batchgenerators.transforms.sample_normalization_transforms import MeanStdNormalizationTransform
    
    transforms = Compose([
        ContrastAugmentationTransform(), # randomly adjust contrast
        MeanStdNormalizationTransform(mean=[img.mean()], std=[img.std()])]) # use concrete values since we only have one sample (have to estimate it over whole dataset otherwise)

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
it. We will therfore use the already implemented ``UNet3dPytorch``:

.. code:: ipython3

    import warnings
    warnings.simplefilter("ignore", UserWarning) # ignore UserWarnings raised by dependency code
    warnings.simplefilter("ignore", FutureWarning) # ignore FutureWarnings raised by dependency code
    
    
    from delira.training import PyTorchExperiment
    from delira.training.train_utils import create_optims_default_pytorch
    from delira.models.segmentation import UNet3dPyTorch
    
    logger.info("Init Experiment")
    experiment = PyTorchExperiment(params, UNet3dPyTorch,
                                   name="Segmentation3dExample",
                                   save_path="./tmp/delira_Experiments",
                                   optim_builder=create_optims_default_pytorch,
                                   gpu_ids=[0], mixed_precision=True)
    experiment.save()
    
    model = experiment.run(manager_train, manager_val)

See Also
--------

For a more detailed explanation have a look at \* `the introduction
tutorial <tutorial_delira.ipynb,>`__ \* `the classification
example <classification_pytorch.ipynb,>`__ \* `the 2d segmentation
example <segmentation_2d_pytorch.ipynb,>`__ \* `the generative
adversarial example <gan_pytorch.ipynb,>`__
