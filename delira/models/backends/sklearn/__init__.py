from delira import get_backends as _get_backends
if "SKLEARN" in _get_backends():
    from delira.models.backends.sklearn.abstract_network import \
        SklearnEstimator
