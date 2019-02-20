import logging

file_logger = logging.getLogger(__name__)

from delira import get_backends

if "TORCH" in get_backends():
    import torch.nn as nn
    import torch.nn.functional as F
    from .classification_network import ClassificationNetworkBasePyTorch

    class VGG3DClassificationNetworkPyTorch(ClassificationNetworkBasePyTorch):
        """
        Exemplaric VGG Network for 3D Classification

        Notes
        -----
        The original network has been adjusted to fit for 3D data

        References
        ----------
        https://arxiv.org/abs/1409.1556

        See Also
        --------
        :class:`ClassificationNetworkBasePyTorch`

        """

        def __init__(self, in_channels: int, n_outputs: int, **kwargs):
            """

            Parameters
            ----------
            in_channels : int
                number of input channels
            n_outputs : int
                number of outputs
            **kwargs :
                additional keyword arguments
            """
            super().__init__(in_channels, n_outputs, **kwargs)

        @staticmethod
        def _build_model(in_channels: int, n_outputs: int, **kwargs):
            """
            Helper Function to build the actual model

            Parameters
            ----------
            in_channels : int
                number of input channels
            n_outputs : int
                number of outputs
            **kwargs :
                additional keyword arguments

            Returns
            -------
            torch.nn.Module
                ensembeled model

            """
            class VGGlike3D(nn.Module):
                def __init__(self, in_channels=3, n_outputs=2):
                    super().__init__()
                    self.conv1 = nn.Conv3d(
                        in_channels, 64, 3, stride=2, padding=0)
                    self.conv2 = nn.Conv3d(64, 64, 3, stride=1, padding=0)
                    self.bn1 = nn.BatchNorm3d(64)

                    self.conv3 = nn.Conv3d(64, 128, 3, stride=2, padding=0)
                    self.conv4 = nn.Conv3d(128, 128, 3, stride=1, padding=0)
                    self.bn2 = nn.BatchNorm3d(128)

                    self.conv5 = nn.Conv3d(128, 256, 3, stride=2, padding=0)
                    self.conv6 = nn.Conv3d(256, 256, (1, 3, 3), stride=1,
                                           padding=0)
                    self.bn3 = nn.BatchNorm3d(256)

                    self.pool = nn.AdaptiveMaxPool3d((1, 16, 16))
                    self.fc1 = nn.Linear(in_features=65536, out_features=1024)
                    self.dropout1 = nn.Dropout(p=0.5, inplace=True)
                    self.fc2 = nn.Linear(1024, 64)
                    self.dropout2 = nn.Dropout(p=0.1, inplace=True)
                    self.fc3 = nn.Linear(64, n_outputs)

                def forward(self, x):
                    x = self.conv1(x)
                    x = self.conv2(x)
                    x = self.bn1(x)
                    x = F.leaky_relu(x, inplace=True)

                    x = self.conv3(x)
                    x = self.conv4(x)
                    x = self.bn2(x)
                    x = F.leaky_relu(x, inplace=True)

                    x = self.conv5(x)
                    x = self.conv6(x)
                    x = self.bn3(x)
                    x = F.leaky_relu(x, inplace=True)

                    x = self.pool(x)
                    x = x.view(x.size(0), -1)
                    x = F.leaky_relu(self.dropout1(self.fc1(x)), inplace=True)
                    x = F.leaky_relu(self.dropout2(self.fc2(x)), inplace=True)
                    x = F.softmax(self.fc3(x), dim=1)
                    return x

            _model = VGGlike3D(in_channels=in_channels, n_outputs=n_outputs)
            return _model
