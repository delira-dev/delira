from delira.io import torch_load_checkpoint, torch_save_checkpoint
from delira.models import AbstractPyTorchNetwork
import torch

class DummyNetwork(AbstractPyTorchNetwork):
    def __init__(self, in_channels, n_outputs):
        super().__init__(in_channels=in_channels, n_outputs=n_outputs)
        self.net = self._build_model(in_channels, n_outputs)

    def forward(self, x):
        return self.module(x)

    @staticmethod
    def _build_model(in_channels, n_outputs):
        return torch.nn.Sequential(
            torch.nn.Linear(in_channels, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, n_outputs)
        )

def test_load_save():
    
    net = DummyNetwork(32, 1)
    torch_save_checkpoint("./model.pt", model=net)
    # fails with weights_only=False only in pytest-mode not in normal execution
    torch_load_checkpoint("./model.pt", weights_only=True)

    torch_save_checkpoint("./model.pt", net, weights_only=True)
    assert torch_load_checkpoint("./model.pt", weights_only=True)

if __name__ == '__main__':
    test_load_save()