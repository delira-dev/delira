from delira import get_backends as _get_backends

if "TF" in _get_backends():
    from delira.models.backends.tf_graph.abstract_network import \
        AbstractTfGraphNetwork
