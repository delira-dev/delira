import unittest
from delira import get_backends
from delira.training import Parameters
from sklearn.metrics import mean_absolute_error
from .utils import create_experiment_test_template_for_backend


if "TORCH" in get_backends():
    from delira.models import AbstractPyTorchNetwork
    import torch

    class DummyNetworkTorch(AbstractPyTorchNetwork):
        def __init__(self):
            super().__init__()

            self.module = torch.nn.Sequential(
                torch.nn.Linear(32, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1)
            )

        def forward(self, x):
            return {
                "pred":
                    self.module(x)
            }


class TestTorchBackend(
    create_experiment_test_template_for_backend("TORCH")
):
    def setUp(self) -> None:
        if "TORCH" in get_backends():
            import torch
            from delira.training import PyTorchExperiment

            params = Parameters(fixed_params={
                "model": {},
                "training": {
                    "losses": {
                        "L1":
                            torch.nn.BCEWithLogitsLoss()},
                    "optimizer_cls": torch.optim.Adam,
                    "optimizer_params": {},
                    "num_epochs": 2,
                    "val_metrics": {"mae": mean_absolute_error},
                    "lr_sched_cls": None,
                    "lr_sched_params": {}}
            })
            model_cls = DummyNetworkTorch
            experiment_cls = PyTorchExperiment

        else:
            params = None
            model_cls = None
            experiment_cls = None

        len_train = 100
        len_test = 50

        self._test_cases = [
            {
                "params": params,
                "network_cls": model_cls,
                "len_train": len_train,
                "len_test": len_test,
                "key_mapping": {"x": "data"},
            }
        ]
        self._experiment_cls = experiment_cls

        super().setUp()


if __name__ == "__main__":
    unittest.main()
