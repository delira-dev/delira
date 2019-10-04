import unittest
import gc
from tests.utils import check_for_tf_graph_backend
from delira.utils import DeliraConfig
from sklearn.metrics import mean_absolute_error
from .utils import create_experiment_test_template_for_backend


if check_for_tf_graph_backend():
    from delira.models import AbstractTfGraphNetwork
    import tensorflow as tf

    class DummyNetworkTfGraph(AbstractTfGraphNetwork):
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

            data = tf.placeholder(shape=[None, 32], dtype=tf.float32)
            labels = tf.placeholder_with_default(
                tf.zeros([tf.shape(data)[0], 1]), shape=[None, 1])

            preds_train = self.model(data)
            preds_eval = self.model(data)

            self.inputs["data"] = data
            self.inputs["label"] = labels
            self.outputs_train["pred"] = preds_train
            self.outputs_eval["pred"] = preds_eval


class TestTfGraphBackend(
    create_experiment_test_template_for_backend("TFGRAPH")
):
    def setUp(self) -> None:
        if check_for_tf_graph_backend():
            import tensorflow as tf
            tf.disable_eager_execution()
            from delira.training import TfGraphExperiment

            config = DeliraConfig()
            config.fixed_params = {
                "model": {},
                "training": {
                    "losses": {
                        "CE":
                            tf.losses.softmax_cross_entropy},
                    "optimizer_cls": tf.train.AdamOptimizer,
                    "optimizer_params": {"learning_rate": 1e-3},
                    "num_epochs": 2,
                    "metrics": {"mae": mean_absolute_error},
                    "lr_sched_cls": None,
                    "lr_sched_params": {}}
            }
            model_cls = DummyNetworkTfGraph
            experiment_cls = TfGraphExperiment

        else:
            config = None
            model_cls = None
            experiment_cls = None

        len_train = 100
        len_test = 50

        self._test_cases = [
            {
                "config": config,
                "network_cls": model_cls,
                "len_train": len_train,
                "len_test": len_test,
                "key_mapping": {"data": "data"},
            }
        ]
        self._experiment_cls = experiment_cls

        super().setUp()

    def tearDown(self):
        import sys
        try:
            del sys.modules["tf"]
        except KeyError:
            pass
        try:
            del tf
        except (UnboundLocalError, NameError):
            pass
        try:
            del sys.modules["tensorflow"]
        except KeyError:
            pass
        try:
            del tensorflow
        except (UnboundLocalError, NameError):
            pass
        gc.collect()


if __name__ == "__main__":
    unittest.main()
