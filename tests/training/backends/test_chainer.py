import unittest
from delira.utils import DeliraConfig
from sklearn.metrics import mean_absolute_error
from .utils import create_experiment_test_template_for_backend

from tests.utils import check_for_chainer_backend


if check_for_chainer_backend():
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


class TestChainerBackend(
    create_experiment_test_template_for_backend("CHAINER")
):
    def setUp(self) -> None:
        if check_for_chainer_backend():
            from delira.training import ChainerExperiment
            import chainer

            config = DeliraConfig()
            config.fixed_params = {
                "model": {},
                "training": {
                    "losses": {
                        "L1":
                            chainer.functions.mean_absolute_error},
                    "optimizer_cls": chainer.optimizers.Adam,
                    "optimizer_params": {},
                    "num_epochs": 2,
                    "metrics": {"mae": mean_absolute_error},
                    "lr_sched_cls": None,
                    "lr_sched_params": {}}
            }
            model_cls = DummyNetworkChainer
            experiment_cls = ChainerExperiment

        else:
            config = None
            model_cls = None
            experiment_cls = None

        len_train = 50
        len_test = 50

        self._test_cases = [
            {
                "config": config,
                "network_cls": model_cls,
                "len_train": len_train,
                "len_test": len_test,
                "key_mapping": {"x": "data"}
            }
        ]
        self._experiment_cls = experiment_cls

        super().setUp()


if __name__ == "__main__":
    unittest.main()
