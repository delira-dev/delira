from .parameters import Parameters
from .experiment import AbstractExperiment
from .abstract_trainer import AbstractNetworkTrainer

import os
if "torch" in os.environ["DELIRA_BACKEND"]:
    from .experiment import PyTorchExperiment
    from .pytorch_trainer import PyTorchNetworkTrainer
    from .metrics import AccuracyMetricPyTorch, AurocMetricPyTorch

if "tf" in os.environ["DELIRA_BACKEND"]:
    from .experiment import TfExperiment
    from .tf_trainer import TfNetworkTrainer
