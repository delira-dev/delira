from delira import get_backends as _get_backends

if "CHAINER" in _get_backends():
    from delira.models.backends.chainer.abstract_network import \
        AbstractChainerNetwork
    from delira.models.backends.chainer.data_parallel import \
        DataParallelChainerNetwork
    from delira.models.backends.chainer.data_parallel import \
        DataParallelChainerOptimizer
    from delira.models.backends.chainer.data_parallel import \
        ParallelOptimizerUpdateModelParameters
    from delira.models.backends.chainer.data_parallel import \
        ParallelOptimizerCumulateGradientsHook
