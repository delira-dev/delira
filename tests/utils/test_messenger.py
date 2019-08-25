from delira.training import BaseExperiment, BaseNetworkTrainer, Parameters
from delira.models import AbstractNetwork
from delira.data_loading import BaseDataManager, AbstractDataset

from delira.utils.messenger import BaseMessenger, SlackMessenger

import unittest
import logging

import numpy as np

logger = logging.getLogger("UnitTestMessenger")


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


class DummyBaseMessenger(BaseMessenger):
    def __init__(
            self,
            experiment,
            notify_epochs=None,
            **kwargs):
        """
        Test messenger for Basemessenger
        """
        super().__init__(experiment, notify_epochs=notify_epochs,
                         **kwargs)

    def emit_message(self, msg):
        logger.info(msg)


class DummyNetwork(AbstractNetwork):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, *args, **kwargs):
        return {}

    @staticmethod
    def closure(model, data_dict: dict, optimizers: dict, losses=None,
                metrics=None, fold=0, **kwargs):
        return {}, {}, {}

    @staticmethod
    def prepare_batch(batch: dict, input_device, output_device):
        return {}


class DummyExperiment(BaseExperiment):
    def __init__(self):
        self.dummy_params = Parameters(fixed_params={
            "model": {},
            "training": {
                "losses": {},
                "optimizer_cls": None,
                "optimizer_params": {},
                "num_epochs": 2,
                "val_metrics": {},
                "lr_sched_cls": None,
                "lr_sched_params": {}}
        })
        super().__init__(self.dummy_params,
                         DummyNetwork,
                         key_mapping={},
                         name="TestExperiment")

    def run(self, raise_error=False, **kwargs):
        if raise_error:
            raise RuntimeError()
        else:
            trainer = self.setup(self.dummy_params, training=True, **kwargs)
            for i in self.dummy_params.nested_get("num_epochs"):
                # execute callbacks
                pass

    def resume(self, *args, raise_error=False, **kwargs):
        if raise_error:
            raise RuntimeError()
        else:
            super().resume(*args, **kwargs)

    def test(self, *args, raise_error=False, **kwargs):
        if raise_error:
            raise RuntimeError()
        else:
            super().test(*args, **kwargs)

    def kfold(self, *args, raise_error=False, **kwargs):
        if raise_error:
            raise RuntimeError()
        else:
            super().kfold(*args, **kwargs)


class TemplateMessengerT(unittest.TestCase):
    def setUp(self) -> None:
        self.msg_run_successful = []
        self.msg_run_failed = []
        # self.msg_resume_successful = []
        # self.msg_resume_failed = []
        self.msg_test_successful = []
        self.msg_test_failed = []
        self.msg_kfold_successful = []
        self.msg_kfold_failed = []

        self.msg_create_experiment = []

        self.messenger_cls = None
        self.messenger_kwargs = {"notify_epochs": 1}

    def create_experiment(self, expected_msg=None):
        with self.assertLogs(logger, level='INFO') as cm:
            dummy_exp = DummyExperiment()
            dummy_exp = self.messenger_cls(dummy_exp, **self.messenger_kwargs)

            if expected_msg is None or not expected_msg:
                logger.info("NoExpectedMessage")

        if expected_msg is None or not expected_msg:
            self.assertEqual(cm.output,
                             ["INFO:UnitTestMessenger:NoExpectedMessage"])
        else:
            self.assertEqual(cm.output, expected_msg)

    def run_experiment(self, raise_error=False, expected_msg=None):
        dummy_exp = DummyExperiment()
        dummy_exp = self.messenger_cls(dummy_exp, **self.messenger_kwargs)

        dset_train = DummyDataset(10)
        dset_test = DummyDataset(10)

        dmgr_train = BaseDataManager(dset_train, 2, 1, None)
        dmgr_test = BaseDataManager(dset_test, 2, 1, None)

        with self.assertLogs(logger, level='INFO') as cm:
            if raise_error:
                self.assertRaises(RuntimeError,
                                  dummy_exp.run(dmgr_train, dmgr_test,
                                                raise_error=True))
            else:
                dummy_exp.run(dmgr_train, dmgr_test, raise_error=False)

            if expected_msg is None or not expected_msg:
                logger.info("NoExpectedMessage")

        if expected_msg is None or not expected_msg:
            self.assertEqual(cm.output,
                             ["INFO:UnitTestMessenger:NoExpectedMessage"])
        else:
            self.assertEqual(cm.output, expected_msg)

    def t_experiment(self, raise_error=False, expected_msg=None):
        dummy_exp = DummyExperiment()
        dummy_exp = self.messenger_cls(dummy_exp, **self.messenger_kwargs)

        dset_test = DummyDataset(10)
        dmgr_test = BaseDataManager(dset_test, 2, 1, None)

        model = DummyNetwork()

        with self.assertLogs(logger, level='INFO') as cm:
            if raise_error:
                self.assertRaises(RuntimeError,
                                  dummy_exp.test(model, dmgr_test, {},
                                                 raise_error=True))
            else:
                dummy_exp.test(model, dmgr_test, {}, raise_error=False)

            if expected_msg is None or not expected_msg:
                logger.info("NoExpectedMessage")

        if expected_msg is None or not expected_msg:
            self.assertEqual(cm.output,
                             ["INFO:UnitTestMessenger:NoExpectedMessage"])
        else:
            self.assertEqual(cm.output, expected_msg)

    def kfold_experiment(self, raise_error=False, expected_msg=None):
        dummy_exp = DummyExperiment()
        dummy_exp = self.messenger_cls(dummy_exp, **self.messenger_kwargs)

        dset = DummyDataset(10)
        dmgr = BaseDataManager(dset, 2, 1, None)

        with self.assertLogs(logger, level='INFO') as cm:
            if raise_error:
                self.assertRaises(RuntimeError,
                                  dummy_exp.kfold(data=dmgr, metrics={},
                                                  num_splits=5,
                                                  raise_error=True))
            else:
                dummy_exp.kfold(data=dmgr, metrics={},
                                num_splits=5, raise_error=False)

            if expected_msg is None:
                logger.info("NoExpectedMessage")

        if expected_msg is None:
            self.assertEqual(cm.output,
                             ["INFO:UnitTestMessenger:NoExpectedMessage"])
        else:
            self.assertEqual(cm.output, expected_msg)
        self.assertEqual(cm.output, self.create_experiment)

    def test_create_experiment(self):
        self.create_experiment(self.msg_create_experiment)

    def test_run_successful(self):
        self.run_experiment(raise_error=False,
                            expected_msg=self.msg_run_successful)

    def test_run_failed(self):
        self.run_experiment(raise_error=True,
                            expected_msg=self.msg_run_failed)

    def test_test_successful(self):
        self.t_experiment(raise_error=False,
                          expected_msg=self.msg_test_successful)

    def test_test_failed(self):
        self.t_experiment(raise_error=True,
                          expected_msg=self.msg_test_failed)

    def test_kfold_successful(self):
        self.kfold_experiment(raise_error=False,
                              expected_msg=self.msg_kfold_successful)

    def test_kfold_failed(self):
        self.kfold_experiment(raise_error=True,
                              expected_msg=self.msg_kfold_failed)


class TestBaseMessenger(TemplateMessengerT):
    def setUp(self) -> None:
        super().setUp()
        self.msg_run_successful = []
        self.msg_run_failed = []
        # self.msg_resume_successful = []
        # self.msg_resume_failed = []
        self.msg_test_successful = []
        self.msg_test_failed = []
        self.msg_kfold_successful = []
        self.msg_kfold_failed = []

        self.msg_create_experiment = []

        self.messenger_cls = DummyBaseMessenger

    def test_create_experiment(self):
        self.create_experiment(self.msg_create_experiment)

    def test_run_successful(self):
        self.run_experiment(raise_error=False,
                            expected_msg=self.msg_run_successful)

    def test_run_failed(self):
        self.run_experiment(raise_error=True,
                            expected_msg=self.msg_run_failed)

    def test_test_successful(self):
        self.t_experiment(raise_error=False,
                          expected_msg=self.msg_test_successful)

    def test_test_failed(self):
        self.t_experiment(raise_error=True,
                          expected_msg=self.msg_test_failed)

    def test_kfold_successful(self):
        self.kfold_experiment(raise_error=False,
                              expected_msg=self.msg_kfold_successful)

    def test_kfold_failed(self):
        self.kfold_experiment(raise_error=True,
                              expected_msg=self.msg_kfold_failed)


if __name__ == '__main__':
    unittest.main()
