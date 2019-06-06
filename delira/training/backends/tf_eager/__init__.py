from delira.training.backends.tf_eager.experiment import TfEagerExperiment
from delira.training.backends.tf_eager.trainer import TfEagerNetworkTrainer
from delira.training.backends.tf_eager.utils import convert_to_numpy \
    as convert_tfeager_to_numpy
from delira.training.backends.tf_eager.utils import switch_tf_execution_mode
from delira.training.backends.tf_eager.utils import create_optims_default \
    as create_tfeager_optims_default
