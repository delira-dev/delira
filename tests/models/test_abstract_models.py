import unittest
import numpy as np
from delira import get_backends
from ..utils import check_for_chainer_backend, check_for_torch_backend, \
    check_for_tf_backend, check_for_sklearn_backend


class TestAbstractModels(unittest.TestCase):

    @staticmethod
    def _setup_torch(*args):
        import torch
        from delira.models.backends.torch import AbstractPyTorchNetwork

        class Model(AbstractPyTorchNetwork):
            def __init__(self):
                super().__init__()
                self.dense = torch.nn.Linear(1, 1)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return {"pred": self.relu(self.dense(x))}

        return Model()

    @staticmethod
    def _setup_torchscript(*args):
        import torch
        from delira.models.backends.torchscript import \
            AbstractTorchScriptNetwork

        class Model(AbstractTorchScriptNetwork):
            def __init__(self):
                super().__init__()
                self.dense = torch.nn.Linear(1, 1)
                self.relu = torch.nn.ReLU()

            @torch.jit.script_method
            def forward(self, x):
                return {"pred": self.relu(self.dense(x))}

        return Model()

    @staticmethod
    def _setup_tfeager(*args):
        import tensorflow as tf
        tf.reset_default_graph()
        from delira.models.backends.tf_eager import AbstractTfEagerNetwork

        class Model(AbstractTfEagerNetwork):
            def __init__(self):
                super().__init__()

                self.dense = tf.keras.layers.Dense(1, activation="relu")

            def call(self, x: tf.Tensor):
                return {"pred": self.dense(x)}

        return Model()

    @staticmethod
    def _setup_tfgraph(*args):
        import tensorflow as tf
        tf.reset_default_graph()
        from delira.models import AbstractTfGraphNetwork
        from delira.training.backends.tf_graph.utils import \
            initialize_uninitialized

        class Model(AbstractTfGraphNetwork):
            def __init__(self):
                super().__init__()
                self.dense = tf.keras.layers.Dense(1, activation="relu")

                data = tf.placeholder(shape=[None, 1],
                                      dtype=tf.float32)

                labels = tf.placeholder_with_default(
                    tf.zeros([tf.shape(data)[0], 1]), shape=[None, 1])

                preds_train = self.dense(data)
                preds_eval = self.dense(data)

                self.inputs["data"] = data
                self.inputs["labels"] = labels
                self.outputs_train["pred"] = preds_train
                self.outputs_eval["pred"] = preds_eval

        model = Model()
        initialize_uninitialized(model._sess)
        return model

    @staticmethod
    def _setup_chainer(*args):
        import chainer
        from delira.models import AbstractChainerNetwork

        class Model(AbstractChainerNetwork):
            def __init__(self):
                super().__init__()

                with self.init_scope():
                    self.dense = chainer.links.Linear(1, 1)

            def forward(self, x):
                return {
                    "pred":
                        chainer.functions.relu(
                            self.dense(x))
                }

        return Model()

    @staticmethod
    def _setup_sklearn(*args):

        from delira.models import SklearnEstimator
        from sklearn.neural_network import MLPRegressor

        class Model(SklearnEstimator):
            def __init__(self):
                # prefit to enable prediction mode afterwards
                module = MLPRegressor()
                module.fit(*args)
                super().__init__(module)

            @staticmethod
            def prepare_batch(batch: dict, input_device, output_device):
                return batch

        return Model()

    def run_model_arg(self, device=None):
        prep_data = self._model.prepare_batch(self._data, input_device=device,
                                              output_device=device)

        pred = self._model(prep_data["data"])
        self.assertIsInstance(pred, dict)

    def run_model_kwarg(self, device=None, keyword="data"):
        prep_data = self._model.prepare_batch(self._data, input_device=device,
                                              output_device=device)

        pred = self._model(**{keyword: prep_data["data"]})
        self.assertIsInstance(pred, dict)

    def setUp(self) -> None:
        self._data = {"data": np.random.rand(100, 1),
                      "label": np.random.rand(100, 1)}

        if "sklearn" in self._testMethodName.lower():
            self._model = self._setup_sklearn(self._data["data"],
                                              self._data["label"])

        elif "chainer" in self._testMethodName.lower():
            self._model = self._setup_chainer()

        elif "pytorch" in self._testMethodName.lower():
            self._model = self._setup_torch()

        elif "torchscript" in self._testMethodName.lower():
            self._model = self._setup_torchscript()

        elif "tf_graph" in self._testMethodName.lower():
            from delira.training.backends.tf_eager import \
                switch_tf_execution_mode

            switch_tf_execution_mode("graph")
            self._model = self._setup_tfgraph()

        elif "tf_eager" in self._testMethodName.lower():
            from delira.training.backends.tf_eager import \
                switch_tf_execution_mode

            switch_tf_execution_mode("eager")
            self._model = self._setup_tfeager()

    @unittest.skipUnless(check_for_sklearn_backend(),
                         "Test should be only executed if sklearn backend is "
                         "installed and specified")
    def test_sklearn(self):
        self.run_model_arg()

    @unittest.skipUnless(check_for_chainer_backend(),
                         "Test should be only executed if chainer backend is "
                         "installed and specified")
    def test_chainer(self):
        import chainer
        self.run_model_arg(chainer.backend.CpuDevice())

    @unittest.skipUnless(check_for_torch_backend(),
                         "Test should be only executed if torch backend is "
                         "installed and specified")
    def test_pytorch(self):
        self.run_model_arg("cpu")

    @unittest.skipUnless(check_for_torch_backend(),
                         "Test should be only executed if torch backend is "
                         "installed and specified")
    def test_torchscript(self):
        self.run_model_arg("cpu")

    @unittest.skipUnless(check_for_tf_backend(),
                         "Test should be only executed if tf backend is "
                         "installed and specified")
    def test_tf_eager(self):
        self.run_model_arg("/cpu:0")

    @unittest.skipUnless(check_for_tf_backend(),
                         "Test should be only executed if tf backend is "
                         "installed and specified")
    def test_tf_graph(self):

        self.run_model_kwarg()


if __name__ == '__main__':
    unittest.main()
