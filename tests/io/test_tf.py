import unittest

from delira import get_backends


class IoTfTest(unittest.TestCase):

    @unittest.skipIf("TF" not in get_backends(),
                     reason="No TF Backend Installed")
    def test_load_save(self):
        from delira.io.tf import load_checkpoint, save_checkpoint
        from delira.models import AbstractTfGraphNetwork
        from delira.training.backends import switch_tf_execution_mode
        from delira.training.backends import initialize_uninitialized

        import tensorflow as tf
        import numpy as np

        switch_tf_execution_mode("graph")

        class DummyNetwork(AbstractTfGraphNetwork):
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

        save_checkpoint("./model", model=net)

        net._sess.run(tf.initializers.global_variables())

        vars_2 = net._sess.run(tf.global_variables())

        load_checkpoint("./model", model=net)

        vars_3 = net._sess.run(tf.global_variables())

        for var_1, var_2 in zip(vars_1, vars_2):
            with self.subTest(var_1=var_1, var2=var_2):
                self.assertTrue(np.all(var_1 != var_2))

        for var_1, var_3 in zip(vars_1, vars_3):
            with self.subTest(var_1=var_1, var_3=var_3):
                self.assertTrue(np.all(var_1 == var_3))

    @unittest.skipIf("TF" not in get_backends(), "No TF Backend installed")
    def test_load_save_eager(self):
        from delira.io.tf import load_checkpoint_eager, save_checkpoint_eager
        from delira.models import AbstractTfEagerNetwork
        from delira.training.backends import switch_tf_execution_mode
        import tensorflow as tf
        import numpy as np

        switch_tf_execution_mode("eager")

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
        input_tensor = np.random.rand(1, 32).astype(np.float32)
        result_pre_save = net()
        save_checkpoint_eager("./model_eager", model=net)

        loaded_state = load_checkpoint_eager("./model_eager", model=net)
        loaded_net = loaded_state["model"]

        result_post_save = loaded_net(input_tensor)

        self.assertTrue(np.array_equal(result_post_save, result_pre_save))


if __name__ == '__main__':
    unittest.main()
