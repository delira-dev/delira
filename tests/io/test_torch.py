import unittest

from delira import get_backends


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
        torch_save_checkpoint("./model_torch.pt", model=net)
        self.assertTrue(torch_load_checkpoint("./model_torch.pt"))

    @unittest.skipIf("TORCH" not in get_backends(),
                     reason="No TORCH Backend Installed")
    def test_jit_save(self):
        from delira.io import jit_save_checkpoint, jit_load_checkpoint
        from delira.models import AbstractPyTorchJITNetwork
        import torch

        class DummyNetwork(AbstractPyTorchJITNetwork):

            def __init__(self):
                super().__init__()
                self.dense = torch.nn.Linear(3, 1)

            @torch.jit.script_method
            def forward(self, x):
                return self.dense(x)

        net = DummyNetwork()
        jit_save_checkpoint("./model_jit.ptj", model=net)
        self.assertTrue(jit_load_checkpoint("./model_jit.ptj"))


if __name__ == '__main__':
    unittest.main()
