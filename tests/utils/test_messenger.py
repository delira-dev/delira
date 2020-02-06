from delira.training import BaseExperiment, BaseNetworkTrainer, Predictor
from delira.utils import DeliraConfig
from delira.models import AbstractNetwork

from delira.data_loading import DataManager

from delira.training.utils import convert_to_numpy_identity


from delira.utils.messenger import BaseMessenger, SlackMessenger

from ..training.backends.utils import DummyDataset

from . import check_for_no_backend

import unittest
import logging
import copy

logger = logging.getLogger("UnitTestMessenger")


class DummyNetwork(AbstractNetwork):
    """
    Emulate Network
    """

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


class DummyTrainer(BaseNetworkTrainer):
    """
    Emulate Trainer states
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.module = DummyNetwork()
        callbacks = kwargs.pop("callbacks", [])
        self._setup(network=self.module, lr_scheduler_cls=None,
                    lr_scheduler_params={}, gpu_ids=[], key_mapping={},
                    convert_batch_to_npy_fn=convert_to_numpy_identity,
                    prepare_batch_fn=self.module.prepare_batch,
                    callbacks=callbacks)

    def train(self, *args, num_epochs=2, **kwargs):
        self._at_training_begin()
        for epoch in range(self.start_epoch, num_epochs + 1):
            self._at_epoch_begin(None, epoch, num_epochs)
            is_best = True if epoch % 2 == 1 else False
            self._at_epoch_end({}, None, epoch, is_best)
        self._at_training_end()
        return DummyNetwork()

    def test(self, *args, **kwargs):
        return [{}], [{}]

    def save_state(self, file_name, *args, **kwargs):
        pass


class DummyPredictor(Predictor):
    """
    Emulate predictor
    """

    def predict(self, *args, **kwargs):
        return {}

    def predict_data_mgr(self, *args, **kwargs):
        yield {}, {}
        return


class DummyExperiment(BaseExperiment):
    def __init__(self):
        dummy_config = DeliraConfig()
        dummy_config.fixed_params = {
            "model": {},
            "training": {
                "losses": {},
                "optimizer_cls": None,
                "optimizer_params": {},
                "num_epochs": 2,
                "lr_sched_cls": None,
                "lr_sched_params": {}}
        }
        super().__init__(dummy_config,
                         DummyNetwork,
                         key_mapping={},
                         name="TestExperiment",
                         trainer_cls=DummyTrainer,
                         predictor_cls=DummyPredictor)

    def run(self, *args, raise_error=False, **kwargs):
        if raise_error:
            raise RuntimeError()
        else:
            return super().run(*args, **kwargs)

    def resume(self, *args, raise_error=False, **kwargs):
        if raise_error:
            raise RuntimeError()
        else:
            return super().resume(*args, **kwargs)

    def test(self, *args, raise_error=False, **kwargs):
        if raise_error:
            raise RuntimeError()
        else:
            return super().test(*args, **kwargs)

    def kfold(self, *args, raise_error=False, **kwargs):
        if raise_error:
            raise RuntimeError()
        else:
            return super().kfold(*args, **kwargs)


class LoggingBaseMessenger(BaseMessenger):
    def __init__(
            self,
            experiment,
            notify_epochs=None,
            **kwargs):
        """
        Test messenger for BaseMessenger
        """
        super().__init__(experiment, notify_epochs=notify_epochs,
                         **kwargs)

    def emit_message(self, msg):
        logger.info(msg)


class TestBaseMessenger(unittest.TestCase):
    def setUp(self) -> None:
        self.msg_run_successful = [
            "INFO:UnitTestMessenger:TestExperiment : Training started.",
            "INFO:UnitTestMessenger:Epoch 1 trained.",
            "INFO:UnitTestMessenger:Epoch 2 trained.",
            "INFO:UnitTestMessenger:TestExperiment : Training completed.",
        ]
        self.msg_run_failed = [
            "INFO:UnitTestMessenger:TestExperiment : Training started.",
            "INFO:UnitTestMessenger:TestExperiment : Training failed. \n",
        ]
        # self.msg_resume_successful = []
        # self.msg_resume_failed = []
        self.msg_test_successful = [
            "INFO:UnitTestMessenger:TestExperiment : Test started.",
            "INFO:UnitTestMessenger:TestExperiment : Test completed.",
        ]
        self.msg_test_failed = [
            "INFO:UnitTestMessenger:TestExperiment : Test started.",
            "INFO:UnitTestMessenger:TestExperiment : Test failed. \n",
        ]
        self.msg_kfold_successful = [
            "INFO:UnitTestMessenger:TestExperiment : Kfold started.",
            "INFO:UnitTestMessenger:Fold 0 started.",
            "INFO:UnitTestMessenger:Epoch 1 trained.",
            "INFO:UnitTestMessenger:Epoch 2 trained.",
            "INFO:UnitTestMessenger:Fold 0 completed.",
            "INFO:UnitTestMessenger:Fold 1 started.",
            "INFO:UnitTestMessenger:Epoch 1 trained.",
            "INFO:UnitTestMessenger:Epoch 2 trained.",
            "INFO:UnitTestMessenger:Fold 1 completed.",
            "INFO:UnitTestMessenger:TestExperiment : Kfold completed.",
        ]
        self.msg_kfold_failed = [
            "INFO:UnitTestMessenger:TestExperiment : Kfold started.",
            "INFO:UnitTestMessenger:TestExperiment : Kfold failed. \n",
        ]

        self.msg_create_experiment = []

        self.messenger_cls = LoggingBaseMessenger
        self.messenger_kwargs = {"notify_epochs": 1}
        self.run_kwargs = {"gpu_ids": [], "logging_type": "tensorboardX",
                           "logging_kwargs": {}, "fold": 3}

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

        dmgr_train = DataManager(dset_train, 2, 1, None)
        dmgr_test = DataManager(dset_test, 2, 1, None)

        with self.assertLogs(logger, level='INFO') as cm:
            if raise_error:
                with self.assertRaises(RuntimeError):
                    dummy_exp.run(dmgr_train, dmgr_test,
                                  raise_error=True, **self.run_kwargs)
            else:
                dummy_exp.run(dmgr_train, dmgr_test, raise_error=False,
                              **self.run_kwargs,)

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
        dmgr_test = DataManager(dset_test, 2, 1, None)

        model = DummyNetwork()

        with self.assertLogs(logger, level='INFO') as cm:
            if raise_error:
                with self.assertRaises(RuntimeError):
                    dummy_exp.test(model, dmgr_test, {},
                                   raise_error=True)
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
        kfold_kwargs = copy.deepcopy(self.run_kwargs)
        kfold_kwargs.pop("fold")

        dummy_exp = DummyExperiment()
        dummy_exp = self.messenger_cls(dummy_exp, **self.messenger_kwargs)

        dset = DummyDataset(10)
        dmgr = DataManager(dset, 2, 1, None)

        with self.assertLogs(logger, level='INFO') as cm:
            if raise_error:
                with self.assertRaises(RuntimeError):
                    dummy_exp.kfold(data=dmgr, metrics={}, num_splits=2,
                                    raise_error=True, **kfold_kwargs)
            else:
                dummy_exp.kfold(data=dmgr, metrics={}, num_splits=2,
                                raise_error=False, **kfold_kwargs)

            if expected_msg is None:
                logger.info("NoExpectedMessage")

        if expected_msg is None:
            self.assertEqual(cm.output,
                             ["INFO:UnitTestMessenger:NoExpectedMessage"])
        else:
            self.assertEqual(cm.output, expected_msg)

    @unittest.skipUnless(
        check_for_no_backend(),
        "Test should only be executed "
        "if no backend is installed")
    def test_create_experiment(self):
        self.create_experiment(self.msg_create_experiment)

    @unittest.skipUnless(
        check_for_no_backend(),
        "Test should only be executed "
        "if no backend is installed")
    def test_run_successful(self):
        self.run_experiment(raise_error=False,
                            expected_msg=self.msg_run_successful)

    @unittest.skipUnless(
        check_for_no_backend(),
        "Test should only be executed "
        "if no backend is installed")
    def test_run_failed(self):
        self.run_experiment(raise_error=True,
                            expected_msg=self.msg_run_failed)

    @unittest.skipUnless(
        check_for_no_backend(),
        "Test should only be executed "
        "if no backend is installed")
    def test_test_successful(self):
        self.t_experiment(raise_error=False,
                          expected_msg=self.msg_test_successful)

    @unittest.skipUnless(
        check_for_no_backend(),
        "Test should only be executed "
        "if no backend is installed")
    def test_test_failed(self):
        self.t_experiment(raise_error=True,
                          expected_msg=self.msg_test_failed)

    @unittest.skipUnless(
        check_for_no_backend(),
        "Test should only be executed "
        "if no backend is installed")
    def test_kfold_successful(self):
        self.kfold_experiment(raise_error=False,
                              expected_msg=self.msg_kfold_successful)

    @unittest.skipUnless(
        check_for_no_backend(),
        "Test should only be executed "
        "if no backend is installed")
    def test_kfold_failed(self):
        self.kfold_experiment(raise_error=True,
                              expected_msg=self.msg_kfold_failed)


class LoggingSlackMessenger(SlackMessenger):
    def emit_message(self, msg):
        logger.info(msg)
        return {}


class TestSlackMessenger(TestBaseMessenger):
    def setUp(self) -> None:
        super().setUp()

        self.msg_create_experiment = [
            "INFO:UnitTestMessenger:Created new experiment: TestExperiment",
        ]

        self.messenger_cls = LoggingSlackMessenger
        self.messenger_kwargs = {"notify_epochs": 1, "token": "dummyToken",
                                 "channel": "dummyChannel"}


if __name__ == '__main__':
    unittest.main()
