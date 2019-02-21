from delira import get_backends

if "TORCH" in get_backends():
    from .classification_network import ClassificationNetworkBasePyTorch
    from .classification_network_3D import VGG3DClassificationNetworkPyTorch

if "TF" in get_backends():
    from .classification_network_tf import ClassificationNetworkBaseTf
    from .ResNet18 import ResNet18
