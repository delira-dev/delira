"""Module Containing Networks for Segmentation Tasks"""
from .unet import UNet2dPyTorch, UNet3dPyTorch

__all__ = [
    'UNet2dPyTorch',
    'UNet3dPyTorch'
]