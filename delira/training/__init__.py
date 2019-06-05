
from .parameters import Parameters
from .experiment import BaseExperiment
from .base_trainer import BaseNetworkTrainer
from .predictor import Predictor

from delira import get_backends

if "TORCH" in get_backends():
    from .experiment import PyTorchExperiment
    from .pytorch_trainer import PyTorchNetworkTrainer

if "TF" in get_backends():
    from .experiment import TfExperiment
    from .tf_trainer import TfNetworkTrainer
