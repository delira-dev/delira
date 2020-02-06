from delira.data_loading import Augmenter, DataLoader, SequentialSampler, \
    AbstractDataset
import numpy as np
from .utils import DummyDataset
from ..utils import check_for_no_backend

import unittest


class TestAugmenters(unittest.TestCase):
    def setUp(self) -> None:
        self._dset_len = 500
        self._batchsize = 3

        if "drop_last" in self._testMethodName:
            self._drop_last = True
        else:
            self._drop_last = False

        dataset = DummyDataset(self._dset_len)
        data_loader = DataLoader(dataset)
        sampler = SequentialSampler.from_dataset(dataset)

        if "parallel" in self._testMethodName:
            self.aug = Augmenter(data_loader, self._batchsize, sampler, 2,
                                 drop_last=self._drop_last)
        else:
            self.aug = Augmenter(data_loader, self._batchsize, sampler, 0,
                                 drop_last=self._drop_last)

    def _aug_test(self):

        num_batches = self._dset_len // self._batchsize
        if not self._drop_last:
            num_batches += int(bool(self._dset_len % self._batchsize))

        last_idx = 0

        for batch in self.aug:
            self.assertIsInstance(batch, dict)

            for v in batch.values():
                # check for batchsize for alll batches except last
                # (which can be smaller)
                if self._drop_last or last_idx < num_batches - 1:
                    self.assertEqual(len(v), self._batchsize)
                else:
                    self.assertLess(len(v), self._batchsize)

            last_idx += 1

        self.assertEqual(last_idx, num_batches)

    # multiple test functions running the same test with different
    # configurations. Must be done in different functions, because
    # configurations are switch based on function name
    @unittest.skipUnless(check_for_no_backend(),
                         "Test should be only executed if no "
                         "backend was installed")
    def test_parallel(self):
        self._aug_test()

    @unittest.skipUnless(check_for_no_backend(),
                         "Test should be only executed if no "
                         "backend was installed")
    def test_parallel_drop_last(self):
        self._aug_test()

    @unittest.skipUnless(check_for_no_backend(),
                         "Test should be only executed if no "
                         "backend was installed")
    def test_sequential(self):
        self._aug_test()

    @unittest.skipUnless(check_for_no_backend(),
                         "Test should be only executed if no "
                         "backend was installed")
    def test_sequential_drop_last(self):
        self._aug_test()

    def _test_sampler_indices(self, parallel: bool):
        class Dataset(AbstractDataset):
            def __init__(self):
                super().__init__(None, None)

                self.data = []

                for i in range(50):
                    self.data.append({"data": i})

            def __getitem__(self, item):
                return self.data[item]

            def __len__(self):
                return 50

        dataset = Dataset()

        data_loader = DataLoader(dataset)
        sampler = SequentialSampler.from_dataset(dataset)

        if parallel:
            aug = Augmenter(data_loader, 1, sampler, 2,
                            drop_last=False)
        else:
            aug = Augmenter(data_loader, 1, sampler, 0,
                            drop_last=False)

        for idx, batch in enumerate(aug):
            self.assertEquals(batch["data"].item(), idx)

    @unittest.skipUnless(check_for_no_backend(),
                         "Test should be only executed if no "
                         "backend was installed")
    def test_sampling_order_parallel(self):
        self._test_sampler_indices(True)

    @unittest.skipUnless(check_for_no_backend(),
                         "Test should be only executed if no "
                         "backend was installed")
    def test_sampling_order_sequential(self):
        self._test_sampler_indices(False)


if __name__ == '__main__':
    unittest.main()
