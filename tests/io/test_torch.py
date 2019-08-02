import unittest

from ..utils import check_for_torch_backend, check_for_torchscript_backend


class IoTorchTest(unittest.TestCase):

    @unittest.skipUnless(check_for_torch_backend(),
                         "Test should be only executed if torch backend is "
                         "installed and specified")
    def test_load_save(self):

        from delira.io.torch import load_checkpoint_torch, \
            save_checkpoint_torch
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
        save_checkpoint_torch("./model_torch.pt", model=net)
        self.assertTrue(load_checkpoint_torch("./model_torch.pt"))

    @unittest.skipUnless(check_for_torchscript_backend(),
                         "Test should be only executed if torch backend is "
                         "installed and specified")
    def test_torchscript_save(self):
        from delira.io.torch import load_checkpoint_torchscript, \
            save_checkpoint_torchscript
        from delira.models import AbstractTorchScriptNetwork
        import torch

        class DummyNetwork(AbstractTorchScriptNetwork):

            def __init__(self):
                super().__init__()
                self.dense = torch.nn.Linear(3, 1)

            @torch.jit.script_method
            def forward(self, x):
                return self.dense(x)

        net = DummyNetwork()
        save_checkpoint_torchscript("./model_jit.ptj", model=net)
        self.assertTrue(load_checkpoint_torchscript("./model_jit.ptj"))


if __name__ == '__main__':
    unittest.main()
