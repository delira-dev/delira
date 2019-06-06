from delira import get_backends as _get_backends

if "TORCH" in _get_backends():
    from delira.models.backends.torch.abstract_network import \
        AbstractPyTorchNetwork
    from delira.models.backends.torch.data_parallel import \
        DataParallelPyTorchNetwork
    from delira.models.backends.torch.utils import scale_loss
