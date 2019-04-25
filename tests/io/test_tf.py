from delira import get_backends
import unittest


class IoTfTest(unittest.TestCase):

    @unittest.skipIf("TF" not in get_backends(),
                     reason="No TF Backend Installed")
    def test_load_save(self):
        from delira.io import tf_load_checkpoint, tf_save_checkpoint
        from delira.models import AbstractTfNetwork
        from delira.training.train_utils import initialize_uninitialized
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
                            64, input_shape=in_channels,
                            bias_initializer='glorot_uniform'),
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.Dense(n_outputs,
                                              bias_initializer='glorot_uniform')
                    ]
                )

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

    if __name__ == '__main__':
        unittest.main()
