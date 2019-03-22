from delira import get_backends
import unittest


class IoTorchTest(unittest.TestCase):

    @unittest.skipIf("TORCH" not in get_backends(),
                     reason="No TORCH Backend Installed")
    def test_load_save(self):

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

        net = DummyNetwork(32, 1)
        torch_save_checkpoint("./model.pt", model=net)
        self.assertTrue(torch_load_checkpoint("./model.pt"))


if __name__ == '__main__':
    unittest.main()
