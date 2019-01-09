from .abstract_network import AbstractNetwork

__all__ = ['AbstractNetwork']
try:
    from .abstract_network import AbstractPyTorchNetwork
    from .classification import VGG3DClassificationNetworkPyTorch, \
        ClassificationNetworkBasePyTorch

    from .segmentation import UNet2dPyTorch, UNet3dPyTorch

    from .gan import GenerativeAdversarialNetworkBasePyTorch

    __all__ += ["AbstractPyTorchNetwork"]

except ImportError as e:
    import warnings
    warnings.warn(ImportWarning(e.msg))
    raise e

finally:
    from .classification import __all__ as __all_clf
    from .gan import __all__ as __all_gan
    from .segmentation import __all__ as __all_seg

__all__ += [
    *__all_clf,
    *__all_gan,
    *__all_seg
]
