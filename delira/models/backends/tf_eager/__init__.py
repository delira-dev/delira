from delira import get_backends as _get_backends

if "TF" in _get_backends():
    from delira.models.backends.tf_eager.abstract_network import \
        AbstractTfEagerNetwork
    from delira.models.backends.tf_eager.data_parallel import \
        DataParallelTfEagerNetwork
