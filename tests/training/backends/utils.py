import numpy as np
from delira.data_loading import AbstractDataset, DataManager
from delira.training import BaseExperiment
from tests.utils import check_for_chainer_backend, \
    check_for_tf_eager_backend, check_for_tf_graph_backend, \
    check_for_sklearn_backend, check_for_torch_backend, \
    check_for_torchscript_backend
import unittest
import logging

from delira.training.callbacks import AbstractCallback

callback_logger = logging.getLogger("CallbackLogger")

_SKIP_CONDITIONS = {
    "CHAINER": check_for_chainer_backend,
    "TFEAGER": check_for_tf_eager_backend,
    "TFGRAPH": check_for_tf_graph_backend,
    "TORCH": check_for_torch_backend,
    "TORCHSCRIPT": check_for_torchscript_backend,
    "SKLEARN": check_for_sklearn_backend
}


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


class LoggingCallback():
    def at_epoch_begin(self, trainer, curr_epoch, **kwargs):
        callback_logger.info("AtEpochBegin_epoch{}".format(curr_epoch))
        return {}

    def at_epoch_end(self, trainer, curr_epoch, **kwargs):
        callback_logger.info("AtEpochEnd_epoch{}".format(curr_epoch))
        return {}

    def at_training_begin(self, trainer, **kwargs):
        callback_logger.info("AtTrainingBegin_fold{}".format(trainer.fold))
        return {}

    def at_training_end(self, trainer, **kwargs):
        callback_logger.info("AtTrainingEnd_fold{}".format(trainer.fold))
        return {}

    def at_iter_begin(self, trainer, iter_num, **kwargs):
        callback_logger.info("AtIterBegin_iter{}".format(iter_num))
        return {}

    def at_iter_end(self, trainer, iter_num, **kwargs):
        callback_logger.info("AtIterEnd_iter{}".format(iter_num))
        return {}


def add_logging_callback(dict_like):
    callbacks = list(dict_like.pop("callbacks", []))
    callbacks.append(LoggingCallback())
    dict_like["callbacks"] = callbacks
    return dict_like


def run_experiment(experiment_cls, config, network_cls, len_train, len_test,
                   **kwargs):
    assert issubclass(experiment_cls, BaseExperiment)
    exp = experiment_cls(config, network_cls, **kwargs)

    dset_train = DummyDataset(len_train)
    dset_test = DummyDataset(len_test)

    dmgr_train = DataManager(dset_train, 16, 4, None)
    dmgr_test = DataManager(dset_test, 16, 1, None)

    return exp.run(dmgr_train, dmgr_test)


def test_experiment(experiment_cls, config, network_cls, len_test, **kwargs):
    assert issubclass(experiment_cls, BaseExperiment)

    exp = experiment_cls(config, network_cls, **kwargs)

    dset_test = DummyDataset(len_test)
    dmgr_test = DataManager(dset_test, 16, 1, None)

    model = network_cls()

    return exp.test(model, dmgr_test, config.nested_get("metrics", {}),
                    kwargs.get("metric_keys", None))


def kfold_experiment(experiment_cls, config, network_cls, len_data,
                     shuffle=True, split_type="random",
                     num_splits=2, val_split=None, **kwargs):
    assert issubclass(experiment_cls, BaseExperiment)

    metric_keys = kwargs.pop("metric_keys", None)

    exp = experiment_cls(config, network_cls, **kwargs)

    dset = DummyDataset(len_data)
    dmgr = DataManager(dset, 16, 1, None)

    return exp.kfold(data=dmgr, metrics=config.nested_get("metrics"),
                     shuffle=shuffle, split_type=split_type,
                     num_splits=num_splits, val_split=val_split,
                     metric_keys=metric_keys)


def create_experiment_test_template_for_backend(backend: str):
    backend_skip = unittest.skipUnless(_SKIP_CONDITIONS[backend](),
                                       "Test should be only executed if "
                                       "backend %s is installed and specified"
                                       % backend)

    class TestCase(unittest.TestCase):

        def setUp(self) -> None:
            # check if the proviced test case hast the following attributes set
            assert hasattr(self, "_experiment_cls")
            assert hasattr(self, "_test_cases")
            self.logging_msg_run = [
                'INFO:CallbackLogger:AtEpochBegin_epoch1',
                'INFO:CallbackLogger:AtEpochEnd_epoch1',
                'INFO:CallbackLogger:AtIterBegin_iter0',
                'INFO:CallbackLogger:AtIterEnd_iter0',
                'INFO:CallbackLogger:AtTrainingBegin_fold0',
                'INFO:CallbackLogger:AtTrainingEnd_fold0',
            ]
            self.logging_msg_test = [
                'INFO:CallbackLogger:AtIterBegin_iter0',
                'INFO:CallbackLogger:AtIterEnd_iter0',
            ]
            self.logging_msg_kfold = [
                'INFO:CallbackLogger:AtEpochBegin_epoch1',
                'INFO:CallbackLogger:AtEpochEnd_epoch1',
                'INFO:CallbackLogger:AtIterBegin_iter0',
                'INFO:CallbackLogger:AtIterEnd_iter0',
                'INFO:CallbackLogger:AtTrainingBegin_fold0',
                'INFO:CallbackLogger:AtTrainingEnd_fold0',
                'INFO:CallbackLogger:AtTrainingBegin_fold1',
                'INFO:CallbackLogger:AtTrainingEnd_fold1',
            ]

        @backend_skip
        def test_experiment_run(self):
            # prototype to run an experiment once for each testcase
            for case in self._test_cases:
                with self.subTest(case=case):
                    case = add_logging_callback(case)
                    with self.assertLogs(callback_logger, "INFO") as cm:
                        run_experiment(self._experiment_cls, **case)

                    for msg in self.logging_msg_run:
                        self.assertIn(msg, cm.output)

        @backend_skip
        def test_experiment_test(self):
            # prototype to test an experiment once with each testcase
            for case in self._test_cases:
                with self.subTest(case=case):
                    _ = case.pop("len_train")
                    case = add_logging_callback(case)
                    with self.assertLogs(callback_logger, "INFO") as cm:
                        test_experiment(self._experiment_cls,
                                        **case)

                    for msg in self.logging_msg_test:
                        self.assertIn(msg, cm.output)

        @backend_skip
        def test_experiment_kfold(self):
            # runs multiple kfolds with each testcase
            # ( 1 for each combination of split_type and val_split)
            for case in self._test_cases:
                with self.subTest(case=case):

                    # combine test and train data to len_data
                    len_data = case.pop("len_test") + case.pop("len_train")
                    case["len_data"] = len_data
                    case = add_logging_callback(case)

                    for split_type in ["random", "stratified", "error"]:
                        with self.subTest(split_type=split_type):

                            if split_type == "error":

                                # must raise ValueError
                                with self.assertRaises(ValueError):
                                    kfold_experiment(
                                        self._experiment_cls, **case,
                                        split_type=split_type, num_splits=2)

                                continue

                            else:
                                for val_split in [0.2, None]:
                                    with self.subTest(val_split=val_split):
                                        with self.assertLogs(
                                                callback_logger, "INFO") as cm:
                                            kfold_experiment(
                                                self._experiment_cls, **case,
                                                val_split=val_split,
                                                split_type=split_type,
                                                num_splits=2,
                                            )

                                        for msg in self.logging_msg_kfold:
                                            self.assertIn(msg, cm.output)

    return TestCase
