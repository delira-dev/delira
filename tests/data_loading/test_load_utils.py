import unittest
from delira.data_loading.load_utils import ensemble_batch
from ..utils import check_for_no_backend
import numpy as np


class LoadUtilsTest(unittest.TestCase):

    def _batch_ensembling(self, add_dim):

        if add_dim:
            ensemble_fn = np.asarray
        else:
            ensemble_fn = np.concatenate
        batch_list = []

        for i in range(50):
            batch_list.append({"data": np.random.rand(1, 2, 3),
                               "test_key": np.random.rand(1, 5)})

        batch_dict = ensemble_batch(batch_list, ensemble_fn=ensemble_fn)
        self.assertListEqual(["data", "test_key"], list(batch_dict.keys()))

        shapes = {"data": (1, 2, 3), "test_key": (1, 5)}
        for key, val in shapes:
            if add_dim:
                shapes[key] = (len(batch_list), *val)
            else:
                shapes[key] = (len(batch_list), *val[1:])

        for key in batch_dict.keys():
            self.assertTupleEqual(batch_dict[key].shape, shapes[key])

    @unittest.skipUnless(check_for_no_backend(),
                         "Test should be only executed if no "
                         "backend was installed")
    def test_batch_ensembling_new_dim(self):
        self._batch_ensembling(True)

    @unittest.skipUnless(check_for_no_backend(),
                         "Test should be only executed if no "
                         "backend was installed")
    def test_batch_ensembling_no_new_dim(self):
        self._batch_ensembling(False)


if __name__ == '__main__':
    unittest.main()
