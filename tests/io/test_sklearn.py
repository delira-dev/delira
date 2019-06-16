import unittest

from ..utils import check_for_sklearn_backend


class IoSklearnTest(unittest.TestCase):

    @unittest.skipUnless(check_for_sklearn_backend(),
                         "Test should be only executed if sklearn backend is "
                         "installed and specified")
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
