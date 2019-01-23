import os
if "torch" in os.environ["DELIRA_BACKEND"]:
    from .classification_network import ClassificationNetworkBasePyTorch
    from .classification_network_3D import VGG3DClassificationNetworkPyTorch
