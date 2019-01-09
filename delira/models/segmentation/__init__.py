"""Module Containing Networks for Segmentation Tasks"""
__all__ = []

try:
    from .unet import UNet2dPyTorch, UNet3dPyTorch

    __all__ += [
        'UNet2dPyTorch',
        'UNet3dPyTorch'
    ]

except ImportError as e:
    import warnings
    warnings.warn(ImportWarning(e.msg))
    raise e
