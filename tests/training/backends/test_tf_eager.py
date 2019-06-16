import unittest
import gc
from tests.utils import check_for_tf_backend
from delira.training import Parameters
from sklearn.metrics import mean_absolute_error
from .utils import create_experiment_test_template_for_backend


if check_for_tf_backend():
    from delira.models import AbstractTfEagerNetwork
    import tensorflow as tf

    class DummyNetworkTfEager(AbstractTfEagerNetwork):
        def __init__(self):
            super().__init__()

            self.model = tf.keras.models.Sequential(
                layers=[
                    tf.keras.layers.Dense(64, input_shape=(
                        32,), bias_initializer='glorot_uniform'),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Dense(
                        1,
                        bias_initializer='glorot_uniform')]
            )

        def call(self, x: tf.Tensor):
            return {"pred": self.model(x)}


class TestTfEagerBackend(
    create_experiment_test_template_for_backend("TF")
):
    def setUp(self) -> None:
        if check_for_tf_backend():
            import tensorflow as tf
            from delira.training import TfEagerExperiment
            from delira.training.backends import switch_tf_execution_mode

            switch_tf_execution_mode("eager")

            params = Parameters(fixed_params={
                "model": {},
                "training": {
                    "losses": {
                        "CE":
                            tf.losses.softmax_cross_entropy},
                    "optimizer_cls": tf.train.AdamOptimizer,
                    "optimizer_params": {"learning_rate": 1e-3},
                    "num_epochs": 2,
                    "val_metrics": {"mae": mean_absolute_error},
                    "lr_sched_cls": None,
                    "lr_sched_params": {}}
            })
            model_cls = DummyNetworkTfEager
            experiment_cls = TfEagerExperiment

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

    def tearDown(self):
        try:
            del tf
        except (UnboundLocalError, NameError):
            pass

        try:
            del tensorflow
        except (UnboundLocalError, NameError):
            pass
        gc.collect()


if __name__ == "__main__":
    unittest.main()
