from delira import get_backends as _get_backends

if "TORCH" in _get_backends():
    from .abstract_network import AbstractTorchScriptNetwork
