import unittest
import numpy as np
from tests.utils import check_for_sklearn_backend
from delira.utils import DeliraConfig
from sklearn.metrics import mean_absolute_error
from .utils import create_experiment_test_template_for_backend, DummyDataset


class TestSklearnBackend(
    create_experiment_test_template_for_backend("SKLEARN")
):
    def setUp(self) -> None:
        if check_for_sklearn_backend():
            from delira.training import SklearnExperiment
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.neural_network import MLPClassifier

            config = DeliraConfig()
            config.fixed_params = {
                "model": {},
                "training": {
                    "losses": {
                        "L1":
                            mean_absolute_error},
                    "optimizer_cls": None,
                    "optimizer_params": {},
                    "num_epochs": 2,
                    "metrics": {"mae": mean_absolute_error},
                    "lr_sched_cls": None,
                    "lr_sched_params": {}}
            }

            # run tests for estimator with and without partial_fit
            model_cls = [
                DecisionTreeClassifier,
                MLPClassifier
            ]

            experiment_cls = SklearnExperiment

        else:
            config = None
            model_cls = []
            experiment_cls = None

        len_train = 50
        len_test = 50

        self._test_cases = [
            {
                "config": config,
                "network_cls": _cls,
                "len_train": len_train,
                "len_test": len_test,
                "key_mapping": {"X": "X"},
                "metric_keys": {"L1": ("pred", "y"),
                                "mae": ("pred", "y")}
            } for _cls in model_cls
        ]
        self._experiment_cls = experiment_cls

        super().setUp()

    @unittest.skipUnless(check_for_sklearn_backend(),
                         "Test should only be executed if SKLEARN backend is "
                         "installed and specified")
    def test_experiment_test(self):
        from delira.data_loading import DataManager

        # iterate over test cases
        for case in self._test_cases:
            with self.subTest(case=case):

                # pop arguments (to use remaining case as kwargs later)
                _ = case.pop("len_train")
                config = case.pop("config")
                metric_keys = case.pop("metric_keys")
                network_cls = case.pop("network_cls")
                len_test = case.pop("len_test")
                exp = self._experiment_cls(config, network_cls, **case)

                # create data
                dset_test = DummyDataset(len_test)
                dmgr_test = DataManager(dset_test, 16, 1, None)

                model = network_cls()

                # must fit on 2 samples to initialize coefficients
                model.fit(np.random.rand(2, 32), np.array([[0], [1]]))

                exp.test(model, dmgr_test,
                         config.nested_get("metrics", {}),
                         metric_keys)


if __name__ == "__main__":
    unittest.main()
