import torch
from delira.training import Parameters
from delira.training.metrics import AurocMetric

params = Parameters(fixed_params={
    "model": {
        "in_channels": 1,
        "n_outputs": 10
    },
    "training": {
        "batch_size": 256, # batchsize to use
        "num_epochs": 10, # number of epochs to train
        "optimizer_cls": torch.optim.Adam, # optimization algorithm to use
        "optimizer_params": {'lr': 1e-3}, # initialization parameters for this algorithm
        "losses": {"CE": torch.nn.CrossEntropyLoss()}, # the loss function
        "lr_sched_cls": None,  # the learning rate scheduling algorithm to use
        "lr_sched_params": {},  # the corresponding initialization parameters
        "metrics": {},
        "val_dataset_metrics": {'AUC': AurocMetric(classes=range(10))},
        # and some evaluation metrics
    }
})

# from trixi.logger import PytorchVisdomLogger
from delira.logging import TrixiHandler
import logging

logger_kwargs = {
    'name': 'ClassificationExampleLogger', # name of our logging environment
    'port': 9999 # port on which our visdom server is alive
}

# logger_cls = PytorchVisdomLogger
#
# # configure logging module (and root logger)
# logging.basicConfig(level=logging.INFO,
#                     handlers=[TrixiHandler(logger_cls, **logger_kwargs)])


# derive logger from root logger
# (don't do `logger = logging.Logger("...")` since this will create a new
# logger which is unrelated to the root logger
logger = logging.getLogger("Test Logger")

from delira.data_loading import TorchvisionClassificationDataset

dataset_train = TorchvisionClassificationDataset("mnist", # which dataset to use
                                                 train=True, # use trainset
                                                 img_shape=(224, 224) # resample to 224 x 224 pixels
                                                )
dataset_val = TorchvisionClassificationDataset("mnist",
                                               train=False,
                                               img_shape=(224, 224)
                                              )

from batchgenerators.transforms import RandomCropTransform, \
                                        ContrastAugmentationTransform, Compose
from batchgenerators.transforms.spatial_transforms import ResizeTransform
from batchgenerators.transforms.sample_normalization_transforms import MeanStdNormalizationTransform

transforms = Compose([
    RandomCropTransform((200, 200)), # Perform Random Crops of Size 200 x 200 pixels
    ResizeTransform((224, 224)), # Resample these crops back to 224 x 224 pixels
    ContrastAugmentationTransform(), # randomly adjust contrast
    MeanStdNormalizationTransform(mean=[0.5], std=[0.5])])

from delira.data_loading import BaseDataManager, SequentialSampler, RandomSampler

manager_train = BaseDataManager(dataset_train, params.nested_get("batch_size"),
                                transforms=transforms,
                                sampler_cls=RandomSampler,
                                n_process_augmentation=4)

manager_val = BaseDataManager(dataset_val, params.nested_get("batch_size"),
                              transforms=transforms,
                              sampler_cls=SequentialSampler,
                              n_process_augmentation=4)

import warnings
warnings.simplefilter("ignore", UserWarning) # ignore UserWarnings raised by dependency code
warnings.simplefilter("ignore", FutureWarning) # ignore FutureWarnings raised by dependency code


from delira.training import PyTorchExperiment
from delira.training.train_utils import create_optims_default_pytorch
from delira.models.classification import ClassificationNetworkBasePyTorch

logger.info("Init Experiment")
experiment = PyTorchExperiment(params, ClassificationNetworkBasePyTorch,
                               name="ClassificationExample",
                               save_path="/home/micha-linux/Desktop/delira_Experiments",
                               optim_builder=create_optims_default_pytorch,
                               gpu_ids=[0])
experiment.save()

model = experiment.run(manager_train, manager_val)

import numpy as np
from tqdm.auto import tqdm  # utility for progress bars

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # set device (use GPU if available)
model = model.to(device)  # push model to device
preds, labels = [], []

with torch.no_grad():
    for i in tqdm(range(len(dataset_val))):
        img = dataset_val[i]["data"]  # get image from current batch
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(device).to(
            torch.float)  # create a tensor from image, push it to device and add batch dimension
        pred_tensor = model(img_tensor)  # feed it through the network
        pred = pred_tensor.argmax(1).item()  # get index with maximum class confidence
        label = np.asscalar(dataset_val[i]["label"])  # get label from batch
        if i % 1000 == 0:
            print("Prediction: %d \t label: %d" % (pred, label))  # print result
        preds.append(pred)
        labels.append(label)

# calculate accuracy
accuracy = (np.asarray(preds) == np.asarray(labels)).sum() / len(preds)
print("Accuracy: %.3f" % accuracy)