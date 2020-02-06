import unittest

import numpy as np

from delira.data_loading import DataManager

from delira.data_loading.data_manager import Augmenter
from ..utils import check_for_no_backend
from .utils import DummyDataset


class DataManagerTest(unittest.TestCase):

    @unittest.skipUnless(check_for_no_backend(),
                         "Test should be only executed if no "
                         "backend was installed")
    def test_datamanager(self):

        batch_size = 16

        np.random.seed(1)
        dset = DummyDataset(600, [0.5, 0.3, 0.2])

        manager = DataManager(dset, batch_size, n_process_augmentation=0,
                              transforms=None)

        self.assertIsInstance(manager.get_batchgen(), Augmenter)

        # create batch manually
        data, labels = [], []
        for i in range(batch_size):
            data.append(dset[i]["data"])
            labels.append(dset[i]["label"])

        batch_dict = {"data": np.asarray(data), "label": np.asarray(labels)}

        augmenter = manager.get_batchgen()
        augmenter_iter = iter(augmenter)
        for key, val in next(augmenter_iter).items():
            self.assertTrue((val == batch_dict[key]).all())

        for key, val in next(augmenter_iter).items():
            self.assertEqual(len(val), batch_size)


if __name__ == '__main__':
    unittest.main()
