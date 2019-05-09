from delira.data_loading import BaseDataLoader, SequentialSampler

import numpy as np
from . import DummyDataset
import unittest


class DataLoaderTest(unittest.TestCase):

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
            len(set([_tmp for _tmp in loader.generate_train_batch()["label"]])),
            1)



if __name__ == '__main__':
    unittest.main()
