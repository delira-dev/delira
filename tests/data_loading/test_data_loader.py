import unittest
from delira.data_loading import DataLoader, SequentialSampler, BatchSampler
import numpy as np
from multiprocessing import Queue
from .utils import DummyDataset
from ..utils import check_for_no_backend


class DataLoaderTest(unittest.TestCase):

    @unittest.skipUnless(check_for_no_backend(),
                         "Test should be only executed if no "
                         "backend was installed")
    def test_data_loader(self):
        dset = DummyDataset(600, [0.5, 0.3, 0.2])
        sampler = SequentialSampler.from_dataset(dset)
        loader = DataLoader(dset)

        batch_sampler = BatchSampler(sampler, 16)
        sampler_iter = iter(batch_sampler)

        self.assertIsInstance(loader(next(sampler_iter)), dict)

        for key, val in loader(next(sampler_iter)).items():
            self.assertEqual(len(val), 16)

        self.assertIn("label", loader(next(sampler_iter)))
        self.assertIn("data", loader(next(sampler_iter)))

        self.assertEqual(
            len(set([_tmp
                     for _tmp in loader(next(sampler_iter))["label"]])),
            1)


if __name__ == '__main__':
    unittest.main()
