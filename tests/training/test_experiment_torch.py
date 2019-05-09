
from delira import get_backends
import os
import delira
import unittest

import numpy as np
from delira.training.metrics import SklearnAccuracyScore
from functools import partial


class TorchExperimentTest(unittest.TestCase):

    @unittest.skipIf("TORCH" not in delira.get_backends(),
                     reason="No TORCH Backend installed")
    def test_experiment(self):

        from delira.training import PyTorchExperiment, Parameters
        from delira.training.callbacks import ReduceLROnPlateauCallbackPyTorch
        from delira.models.classification import ClassificationNetworkBasePyTorch
        from delira.data_loading import AbstractDataset, BaseDataManager
        import torch

        test_cases = [
            (
                Parameters(fixed_params={
                    "model": {},
                    "training": {
                        "losses": {"CE":
                                       torch.nn.BCEWithLogitsLoss()},
                        "optimizer_cls": torch.optim.Adam,
                        "optimizer_params": {"lr": 1e-3},
                        "num_epochs": 2,
                        "val_metrics": {"accuracy": SklearnAccuracyScore(
                            gt_logits=False, pred_logits=True)},
                        "lr_sched_cls": ReduceLROnPlateauCallbackPyTorch,
                        "lr_sched_params": {"mode": "max"}
                    }
                }
                ),
                500,
                50,
                "accuracy",
                "highest"
            )
        ]

        class DummyNetwork(ClassificationNetworkBasePyTorch):

            def __init__(self):
                super().__init__(32, 1)

            def forward(self, x):
                return self.module(x)

            @staticmethod
            def _build_model(in_channels, n_outputs):
                return torch.nn.Sequential(
                    torch.nn.Linear(in_channels, 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, n_outputs)
                )

            @staticmethod
            def prepare_batch(batch_dict, input_device, output_device):
                return {"data": torch.from_numpy(batch_dict["data"]
                                                 ).to(input_device,
                                                      torch.float),
                        "label": torch.from_numpy(batch_dict["label"]
                                                  ).to(output_device,
                                                       torch.float)}

        class DummyDataset(AbstractDataset):
            def __init__(self, length):
                super().__init__(None, None)
                self.length = length

            def __getitem__(self, index):
                return {"data": np.random.rand(32),
                        "label": np.random.randint(0, 1, 1)}

            def __len__(self):
                return self.length

            def get_sample_from_index(self, index):
                return self.__getitem__(index)

        for case in test_cases:
            with self.subTest(case=case):

                params, dataset_length_train, dataset_length_test, \
                    val_score_key, val_score_mode = case

                exp = PyTorchExperiment(params, DummyNetwork,
                                        val_score_key=val_score_key,
                                        val_score_mode=val_score_mode)

                dset_train = DummyDataset(dataset_length_train)
                dset_test = DummyDataset(dataset_length_test)

                dmgr_train = BaseDataManager(dset_train, 16, 4, None)
                dmgr_test = BaseDataManager(dset_test, 16, 1, None)

                net = exp.run(dmgr_train, dmgr_test)
                exp.test(params=params,
                         network=net,
                         datamgr_test=dmgr_test,
                         metrics=params.nested_get("val_metrics"),
                         prepare_batch=partial(net.prepare_batch,
                                               input_device="cpu",
                                               output_device="cpu"))

                exp.kfold(2, dmgr_train, num_splits=2)
                exp.stratified_kfold(2, dmgr_train, num_splits=2)
                exp.stratified_kfold_predict(2, dmgr_train, num_splits=2)


if __name__ == '__main__':
    unittest.main()

