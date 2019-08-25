
from delira.data_loading import AbstractDataset
from delira.training import Parameters
from delira.utils import BaseMessenger
from delira import get_backends
import unittest
from ..training.backends.utils import run_experiment, test_experiment, \
    kfold_experiment

import numpy as np
from functools import partial
import logging
logger = logging.getLogger(__name__)


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


class TestMessenger(BaseMessenger):
    """
    Overwrite emit message because no slack client present
    """

    def __init__(
            self,
            experiment,
            notify_epochs=None,
            **kwargs):
        super().__init__(experiment, notify_epochs=notify_epochs,
                         **kwargs)

    def emit_message(self, msg):
        logging.info(msg)


class ExperimentTest(unittest.TestCase):

    def setUp(self) -> None:
        test_cases_torch = []
        from sklearn.metrics import mean_absolute_error

        # setup torch testcases
        if "TORCH" in get_backends():
            import torch
            from delira.models.classification import \
                ClassificationNetworkBasePyTorch
            from delira.training.callbacks import \
                ReduceLROnPlateauCallbackPyTorch

            class DummyNetworkTorch(ClassificationNetworkBasePyTorch):

                def __init__(self):
                    super().__init__(32, 1)

                def forward(self, x):
                    return {"pred": self.module(x)}

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

            test_cases_torch.append((
                Parameters(fixed_params={
                    "model": {},
                    "training": {
                        "losses": {"CE":
                                   torch.nn.BCEWithLogitsLoss()},
                        "optimizer_cls": torch.optim.Adam,
                        "optimizer_params": {"lr": 1e-3},
                        "num_epochs": 2,
                        "val_metrics": {"val_mae": mean_absolute_error},
                        "lr_sched_cls": ReduceLROnPlateauCallbackPyTorch,
                        "lr_sched_params": {"mode": "min"}
                    }
                }
                ),
                500,
                50,
                "mae",
                "lowest",
                DummyNetworkTorch))

            self._test_cases_torch = test_cases_torch
        logger.info(self._testMethodName)

    @unittest.skipIf("TORCH" not in get_backends(),
                     reason="No TORCH Backend installed")
    def test_experiment_run_torch(self):

        from delira.training import PyTorchExperiment
        from delira.data_loading import BaseDataManager

        for case in self._test_cases_torch:
            with self.subTest(case=case):

                (params, dataset_length_train, dataset_length_test,
                 val_score_key, val_score_mode, network_cls) = case

                exp = PyTorchExperiment(params, network_cls,
                                        key_mapping={"x": "data"},
                                        val_score_key=val_score_key,
                                        val_score_mode=val_score_mode)

                with self.assertLogs(level='ERROR') as log:
                    exp = TestMessenger(exp, "token_here",
                                        "channel_here",
                                        notify_epochs=1)
                    self.assertIn('Slack message was not emitted correctly!',
                                  log.output[0])

                dset_train = DummyDataset(dataset_length_train)
                dset_test = DummyDataset(dataset_length_test)

                dmgr_train = BaseDataManager(dset_train, 4, 1, None)
                dmgr_test = BaseDataManager(dset_test, 4, 1, None)

                with self.assertLogs(level='INFO') as log:
                    exp.run(dmgr_train, dmgr_test)
                    self.assertIn('Training started.', log.output[0])
                    # self.assertIn('Epoch 1 trained.', log.output[1])
                    # self.assertIn('Epoch 2 trained.', log.output[2])
                    # self.assertIn('Training completed.', log.output[3])

    @unittest.skipIf("TORCH" not in get_backends(),
                     reason="No TORCH Backend installed")
    def test_experiment_test_torch(self):
        from delira.training import PyTorchExperiment
        from delira.data_loading import BaseDataManager

        for case in self._test_cases_torch:
            with self.subTest(case=case):
                (params, dataset_length_train, dataset_length_test,
                 val_score_key, val_score_mode, network_cls) = case

                exp = PyTorchExperiment(params, network_cls,
                                        key_mapping={"x": "data"},
                                        val_score_key=val_score_key,
                                        val_score_mode=val_score_mode)

                with self.assertLogs(level='ERROR') as log:
                    exp = TestMessenger(exp, "token_here",
                                        "channel_here",
                                        notify_epochs=1)
                    self.assertIn('Slack message was not emitted correctly!',
                                  log.output[0])

                model = network_cls()

                dset_test = DummyDataset(dataset_length_test)
                dmgr_test = BaseDataManager(dset_test, 16, 1, None)

                prepare_batch = partial(
                    model.prepare_batch,
                    output_device="cpu",
                    input_device="cpu")

                with self.assertLogs(level='INFO') as log:
                    exp.test(model, dmgr_test,
                             params.nested_get("val_metrics"),
                             prepare_batch=prepare_batch)
                    self.assertIn('Test started.', log.output[0])

    @unittest.skipIf("TORCH" not in get_backends(),
                     reason="No TORCH Backend installed")
    def test_experiment_kfold_torch(self):
        from delira.training import PyTorchExperiment
        from delira.data_loading import BaseDataManager
        from copy import deepcopy

        # all test cases
        for case in self._test_cases_torch:
            with self.subTest(case=case):
                (params, dataset_length_train,
                 dataset_length_test, val_score_key,
                 val_score_mode, network_cls) = case

                # both split_types
                split_type = "random"
                with self.subTest(split_type="random"):
                    # check all types of validation data
                    val_split = 0.2
                    with self.subTest(val_split=0.2):
                        # disable lr scheduling if no validation data
                        # is present
                        _params = deepcopy(params)
                        if val_split is None:
                            _params["fixed"]["training"
                                             ]["lr_sched_cls"] = None
                        exp = PyTorchExperiment(
                            _params, network_cls,
                            key_mapping={"x": "data"},
                            val_score_key=val_score_key,
                            val_score_mode=val_score_mode)

                        with self.assertLogs(level='ERROR') as log:
                            exp = TestMessenger(exp, "token_here",
                                                "channel_here",
                                                notify_epochs=1)
                            self.assertIn(
                                'Slack message was not emitted correctly!',
                                log.output[0])

                    dset = DummyDataset(
                        dataset_length_test + dataset_length_train)

                    dmgr = BaseDataManager(dset, 16, 1, None)
                    with self.assertLogs(level='INFO') as log:
                        exp.kfold(
                            dmgr,
                            params.nested_get("val_metrics"),
                            shuffle=True,
                            split_type=split_type,
                            val_split=val_split,
                            num_splits=2)
                        self.assertIn('Kfold started.', log.output[0])


if __name__ == '__main__':
    unittest.main()
