import unittest
import numpy as np
from delira import get_backends
from delira.training import Parameters
from sklearn.metrics import mean_absolute_error
from .utils import create_experiment_test_template_for_backend, DummyDataset


if "SKLEARN" in get_backends():
    from delira.models import AbstractChainerNetwork
    import chainer

    # define this outside, because it has to be pickleable, which it won't be,
    # wehn defined inside a function
    class DummyNetworkChainer(AbstractChainerNetwork):
        def __init__(self):
            super().__init__()

            with self.init_scope():
                self.dense_1 = chainer.links.Linear(32, 64)
                self.dense_2 = chainer.links.Linear(64, 1)

        def forward(self, x):
            return {
                "pred":
                    self.dense_2(chainer.functions.relu(
                        self.dense_1(x)))
            }


class TestSklearnBackend(
    create_experiment_test_template_for_backend("SKLEARN")
):
    def setUp(self) -> None:
        if "SKLEARN" in get_backends():
            from delira.training import SklearnExperiment
            import sklearn

            params = Parameters(fixed_params={
                "model": {},
                "training": {
                    "losses": {
                        "L1":
                            mean_absolute_error},
                    "optimizer_cls": None,
                    "optimizer_params": {},
                    "num_epochs": 2,
                    "val_metrics": {"mae": mean_absolute_error},
                    "lr_sched_cls": None,
                    "lr_sched_params": {}}
            })

            # run tests for estimator with and without partial_fit
            model_cls = [sklearn.tree.DecisionTreeClassifier,
                         sklearn.neural_network.MLPClassifier]

            experiment_cls = SklearnExperiment

        else:
            params = None
            model_cls = []
            experiment_cls = None

        len_train = 50
        len_test = 50

        self._test_cases = [
            {
                "params": params,
                "network_cls": _cls,
                "len_train": len_train,
                "len_test": len_test,
                "key_mapping": {"x": "data"}
            } for _cls in model_cls
        ]
        self._experiment_cls = experiment_cls

        super().setUp()

    @unittest.skipIf("SKLEARN" not in get_backends(),
                     reason="No SKLEARN backend installed")
    def test_experiment_test(self):
        from delira.data_loading import BaseDataManager

        # iterate over test cases
        for case in self._test_cases:
            with self.subTest(case=case):

                # pop arguments (to use remaining case as kwargs later)
                _ = case.pop("len_train")
                params = case.pop("params")
                network_cls = case.pop("network_cls")
                len_test = case.pop("len_test")
                exp = self._experiment_cls(case["params"], network_cls, **case)

                # create data
                dset_test = DummyDataset(len_test)
                dmgr_test = BaseDataManager(dset_test, 16, 1, None)

                model = network_cls()

                # must fit on 2 samples to initialize coefficients
                model.fit(np.random.rand(2, 32), np.array([[0], [1]]))

            return exp.test(model, dmgr_test,
                            params.nested_get("val_metrics", {}))


if __name__ == "__main__":
    unittest.main()