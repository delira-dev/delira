import unittest
from delira import get_backends
from copy import deepcopy
import numpy as np


class TestDataParallelTorch(unittest.TestCase):

    def setUp(self) -> None:
        if "TORCH" in get_backends():
            from delira.models.backends.torch import AbstractPyTorchNetwork, \
                DataParallelPyTorchNetwork
            import torch

            class SimpleModel(AbstractPyTorchNetwork):
                def __init__(self):
                    super().__init__()

                    self.dense_1 = torch.nn.Linear(3, 32)
                    self.dense_2 = torch.nn.Linear(32, 2)
                    self.relu = torch.nn.ReLU()

                def forward(self, x):
                    return {"pred": self.dense_2(self.relu(self.dense1(x)))}

            model = SimpleModel()

            self.optimizer = torch.nn.Adam(model.parameters())

            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                self.model = DataParallelPyTorchNetwork(model, [0, 1])
            else:
                self.model = model

    @unittest.skipIf("TORCH" not in get_backends(),
                     "No TORCH Backend installed")
    def test_update(self):
        import torch

        input_tensor = torch.rand(10, 3)
        label_tensor = torch.rand(10, 2)

        model_copy = deepcopy(self.model)

        preds = self.model(input_tensor)

        loss = (preds["pred"] - label_tensor).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for orig_param, updated_param in zip(model_copy.parameters(),
                                             self.model.parameters()):
            self.assertFalse(np.array_equal(orig_param.detach().cpu().numpy(),
                                            updated_param.detach().cpu().numpy()
                                            ))

if __name__ == '__main__':
    unittest.main()