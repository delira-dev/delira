import unittest

import numpy as np

from delira.data_loading import BaseDataLoader, SequentialSampler
from .utils import DummyDataset
from ..utils import check_for_no_backend


class DataLoaderTest(unittest.TestCase):

    @unittest.skipUnless(check_for_no_backend(),
                         "Test should be only executed if no "
                         "backend was installed")
    def test_data_loader(self):
        np.random.seed(1)
        dset = DummyDataset(600, [0.5, 0.3, 0.2])
        sampler = SequentialSampler.from_dataset(dset)
        loader = BaseDataLoader(dset, batch_size=16, sampler=sampler)

        self.assertIsInstance(loader.generate_train_batch(), dict)

        for key, val in loader.generate_train_batch().items():
            self.assertEqual(len(val), 16)

        self.assertIn("label", loader.generate_train_batch())
        self.assertIn("data", loader.generate_train_batch())

        self.assertEqual(
            len(set([_tmp
                     for _tmp in loader.generate_train_batch()["label"]])),
            1)


if __name__ == '__main__':
    unittest.main()
