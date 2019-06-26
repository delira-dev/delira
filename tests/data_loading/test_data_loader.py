import unittest

from delira.data_loading import DataLoader, SequentialSampler
from . import DummyDataset


class DataLoaderTest(unittest.TestCase):

    def test_data_loader(self):
        dset = DummyDataset(600, [0.5, 0.3, 0.2])
        sampler = SequentialSampler.from_dataset(dset)
        loader = DataLoader(dset)

        self.assertIsInstance(loader(sampler(16)), dict)

        for key, val in loader(sampler(16)).items():
            self.assertEqual(len(val), 16)

        self.assertIn("label", loader(sampler(16)))
        self.assertIn("data", loader(sampler(16)))

        self.assertEqual(
            len(set([_tmp
                     for _tmp in loader(sampler(16))["label"]])),
            1)


if __name__ == '__main__':
    unittest.main()
