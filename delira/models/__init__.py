from .abstract_network import AbstractNetwork, AbstractPyTorchNetwork
from .classification import VGG3DClassificationNetworkPyTorch, \
    ClassificationNetworkBasePyTorch

from .segmentation import UNet2dPyTorch, UNet3dPyTorch

from .gan import GenerativeAdversarialNetworkBasePyTorch


from .classification import __all__ as __all_clf
from .gan import __all__ as __all_gan
from .segmentation import __all__ as __all_seg

__all__ = [
    'AbstractNetwork',
    'AbstractPyTorchNetwork',
    *__all_clf,
    *__all_gan,
    *__all_seg
]
