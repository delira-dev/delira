import numpy as np
from delira.data_loading import AbstractDataset, BaseDataManager
from delira.training import BaseExperiment
from delira import get_backends
import unittest


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


def run_experiment(experiment_cls, params, network_cls, len_train, len_test,
                   **kwargs):

    assert issubclass(experiment_cls, BaseExperiment)
    exp = experiment_cls(params, network_cls, **kwargs)

    dset_train = DummyDataset(len_train)
    dset_test = DummyDataset(len_test)

    dmgr_train = BaseDataManager(dset_train, 16, 4, None)
    dmgr_test = BaseDataManager(dset_test, 16, 1, None)

    return exp.run(dmgr_train, dmgr_test)


def test_experiment(experiment_cls, params, network_cls, len_test, **kwargs):
    assert issubclass(experiment_cls, BaseExperiment)

    exp = experiment_cls(params, network_cls, **kwargs)

    dset_test = DummyDataset(len_test)
    dmgr_test = BaseDataManager(dset_test, 16, 1, None)

    model = network_cls()

    return exp.test(model, dmgr_test, params.nested_get("val_metrics", {}))


def kfold_experiment(experiment_cls, params, network_cls, len_data,
                     shuffle=True, split_type="random",
                     num_splits=2, val_split=None, **kwargs):
    assert issubclass(experiment_cls, BaseExperiment)

    exp = experiment_cls(params, network_cls, **kwargs)

    dset = DummyDataset(len_data)
    dmgr = BaseDataManager(dset, 16, 1, None)

    return exp.kfold(data=dmgr, metrics=params.nested_get("val_metrics"),
                     shuffle=shuffle, split_type=split_type,
                     num_splits=num_splits, val_split=val_split)


def create_experiment_test_template_for_backend(backend: str):
    backend_skip = unittest.skipIf(backend not in get_backends(),
                                   reason="No %s backend installed" % backend)

    class TestCase(unittest.TestCase):

        def setUp(self) -> None:
            # check if the proviced test case hast the following attributes set
            assert hasattr(self, "_experiment_cls")
            assert hasattr(self, "_test_cases")

        @backend_skip
        def test_experiment_run(self):
            # prototype to run an experiment once for each testcase
            for case in self._test_cases:
                with self.subTest(case=case):
                    run_experiment(self._experiment_cls, **case)

        @backend_skip
        def test_experiment_test(self):
            # prototype to test an experiment once with each testcase
            for case in self._test_cases:
                with self.subTest(case=case):
                    _ = case.pop("len_train")
                    test_experiment(self._experiment_cls,
                                    **case)

        @backend_skip
        def test_experiment_kfold(self):
            # runs multiple kfolds with each testcase
            # ( 1 for each combination of split_type and val_split)
            for case in self._test_cases:
                with self.subTest(case=case):

                    # combine test and train data to len_data
                    len_data = case.pop("len_test") + case.pop("len_train")
                    case["len_data"] = len_data

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
                                        kfold_experiment(
                                            self._experiment_cls, **case,
                                            val_split=val_split,
                                            split_type=split_type, num_splits=2
                                        )

    return TestCase
