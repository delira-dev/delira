from .abstract_network import AbstractNetwork

from delira import get_backends

if "TORCH" in get_backends():
    from .abstract_network import AbstractPyTorchNetwork
    from .classification import VGG3DClassificationNetworkPyTorch, \
        ClassificationNetworkBasePyTorch

    from .segmentation import UNet2dPyTorch, UNet3dPyTorch

    from .gan import GenerativeAdversarialNetworkBasePyTorch

if "TF" in get_backends():
    from .abstract_network import AbstractTfNetwork
    from .classification import ClassificationNetworkBaseTf