import os
if "torch" in os.environ["DELIRA_BACKEND"]:
    from .generative_adversarial_network import \
        GenerativeAdversarialNetworkBasePyTorch
