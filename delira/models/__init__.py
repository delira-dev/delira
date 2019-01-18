from .abstract_network import AbstractNetwork

import os
if "torch" in os.environ["DELIRA_BACKEND"]:
    from .abstract_network import AbstractPyTorchNetwork
    from .classification import VGG3DClassificationNetworkPyTorch, \
        ClassificationNetworkBasePyTorch

    from .segmentation import UNet2dPyTorch, UNet3dPyTorch

    from .gan import GenerativeAdversarialNetworkBasePyTorch
