import unittest

from delira import get_backends


class IoSklearnTest(unittest.TestCase):

    @unittest.skipIf("SKLEARN" not in get_backends(),
                     reason="No SKLEARN Backend Installed")
    def test_load_save(self):

        from delira.io.sklearn import load_checkpoint, save_checkpoint
        from delira.models import SklearnEstimator
        from sklearn.tree import DecisionTreeRegressor
        import numpy as np

        net = SklearnEstimator(DecisionTreeRegressor())
        net.fit(X=np.random.rand(2, 32), y=np.random.rand(2))
        save_checkpoint("./model_sklearn.pkl", model=net)
        self.assertTrue(load_checkpoint("./model_sklearn.pkl"))


if __name__ == '__main__':
    unittest.main()
