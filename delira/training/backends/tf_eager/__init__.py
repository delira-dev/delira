from delira import get_backends as _get_backends

if "TF" in _get_backends():
    from delira.training.backends.tf_eager.experiment import TfEagerExperiment
    from delira.training.backends.tf_eager.trainer import TfEagerNetworkTrainer
    from delira.training.backends.tf_eager.utils import convert_to_numpy \
        as convert_tfeager_to_numpy
    from delira.training.backends.tf_eager.utils import create_optims_default \
        as create_tfeager_optims_default
