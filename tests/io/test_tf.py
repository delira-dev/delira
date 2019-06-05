import unittest

from delira import get_backends


class IoTfTest(unittest.TestCase):

    @unittest.skipIf("TF" not in get_backends(),
                     reason="No TF Backend Installed")
    def test_load_save(self):
        from delira.io import tf_load_checkpoint, tf_save_checkpoint
        from delira.models import AbstractTfNetwork
        from delira.training.utils import initialize_uninitialized
        import tensorflow as tf
        import numpy as np

        class DummyNetwork(AbstractTfNetwork):
            def __init__(self, in_channels, n_outputs):
                super().__init__(in_channels=in_channels, n_outputs=n_outputs)
                self.net = self._build_model(in_channels, n_outputs)

            @staticmethod
            def _build_model(in_channels, n_outputs):
                return tf.keras.models.Sequential(
                    layers=[
                        tf.keras.layers.Dense(
                            64,
                            input_shape=in_channels,
                            bias_initializer='glorot_uniform'),
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.Dense(
                            n_outputs,
                            bias_initializer='glorot_uniform')])

        net = DummyNetwork((32,), 1)
        initialize_uninitialized(net._sess)

        vars_1 = net._sess.run(tf.global_variables())

        tf_save_checkpoint("./model", model=net)

        net._sess.run(tf.initializers.global_variables())

        vars_2 = net._sess.run(tf.global_variables())

        tf_load_checkpoint("./model", model=net)

        vars_3 = net._sess.run(tf.global_variables())

        for var_1, var_2 in zip(vars_1, vars_2):
            with self.subTest(var_1=var_1, var2=var_2):
                self.assertTrue(np.all(var_1 != var_2))

        for var_1, var_3 in zip(vars_1, vars_3):
            with self.subTest(var_1=var_1, var_3=var_3):
                self.assertTrue(np.all(var_1 == var_3))

    @unittest.skipIf("TF" not in get_backends(), "No TF Backend installed")
    def test_load_save_eager(self):
        from delira.io import tf_eager_load_checkpoint, tf_eager_save_checkpoint
        from delira.models import AbstractTfEagerNetwork
        import tensorflow as tf
        import numpy as np

        if not tf.executing_eagerly():
            tf.enable_eager_execution()

        class DummyNetwork(AbstractTfEagerNetwork):
            def __init__(self, in_channels, n_outputs):
                super().__init__(in_channels=in_channels, n_outputs=n_outputs)
                self.net = self._build_model(in_channels, n_outputs)

            @staticmethod
            def _build_model(in_channels, n_outputs):
                return tf.keras.models.Sequential(
                    layers=[
                        tf.keras.layers.Dense(
                            64,
                            input_shape=in_channels,
                            bias_initializer='glorot_uniform'),
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.Dense(
                            n_outputs,
                            bias_initializer='glorot_uniform')])

            def call(self, inputs):
                return self.net(inputs)

        net = DummyNetwork((32,), 1)
        input_tensor = np.random.rand(1, 32)
        result_pre_save = net(input_tensor)
        tf_eager_save_checkpoint("./model_eager", model=net)

        loaded_state = tf_eager_load_checkpoint("./model_eager", model=net)
        loaded_net = loaded_state["model"]

        result_post_save = loaded_net(input_tensor)

        self.assertTrue(np.array_equal(result_post_save, result_pre_save))

    if __name__ == '__main__':
        unittest.main()
