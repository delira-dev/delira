from .parameters import Parameters
from .experiment import AbstractExperiment
from .abstract_trainer import AbstractNetworkTrainer

import os
if "torch" in os.environ["DELIRA_BACKEND"]:
    from .experiment import PyTorchExperiment
    from .pytorch_trainer import PyTorchNetworkTrainer
    from .metrics import AccuracyMetricPyTorch, AurocMetricPyTorch
