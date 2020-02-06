import unittest
from delira.data_loading import DataLoader, SequentialSampler, BatchSampler
from .utils import DummyDataset
import numpy as np
from ..utils import check_for_no_backend


class DataLoaderTest(unittest.TestCase):

    def _test_data_loader(self, data):
        loader = DataLoader(data)
        sampler = SequentialSampler.from_dataset(loader.dataset)

        batch_sampler = BatchSampler(sampler, 16)
        sampler_iter = iter(batch_sampler)

        self.assertIsInstance(loader(next(sampler_iter)), dict)

        for key, val in loader(next(sampler_iter)).items():
            self.assertEqual(len(val), 16)

        self.assertIn("label", loader(next(sampler_iter)))
        self.assertIn("data", loader(next(sampler_iter)))

        self.assertEquals(loader.process_id, 0)
        loader.process_id = 456
        self.assertEquals(loader.process_id, 456)
        with self.assertRaises(AttributeError):
            loader.process_id = 123

    @unittest.skipUnless(check_for_no_backend(),
                         "Test should be only executed if no "
                         "backend was installed")
    def test_data_loader_dset(self):
        dset = DummyDataset(600, [0.5, 0.3, 0.2])
        self._test_data_loader(dset)

    @unittest.skipUnless(check_for_no_backend(),
                         "Test should be only executed if no "
                         "backend was installed")
    def test_data_loader_dict(self):
        data = {"label": np.random.rand(600),
                "data": np.random.rand(600, 1, 3, 3)}
        self._test_data_loader(data)

    @unittest.skipUnless(check_for_no_backend(),
                         "Test should be only executed if no "
                         "backend was installed")
    def test_data_loader_iterable(self):
        data = [{"label": np.random.rand(1), "data": np.random.rand(1, 3, 3)}
                for i in range(600)]
        self._test_data_loader(data)


if __name__ == '__main__':
    unittest.main()
