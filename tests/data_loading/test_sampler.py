import unittest
import numpy as np
from delira.data_loading.sampler import RandomSamplerWithReplacement, \
    PrevalenceRandomSampler, SequentialSampler, \
    RandomSamplerNoReplacement, BatchSampler, AbstractSampler

from ..utils import check_for_no_backend
from .utils import DummyDataset


class SamplerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dset = DummyDataset(600, [0.5, 0.3, 0.2])

    @unittest.skipUnless(check_for_no_backend(),
                         "Test should only be executed "
                         "if no backend is installed/specified")
    def test_batch_sampler(self):
        for batchsize in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            for truncate in [True, False]:

                with self.subTest(batchsize=batchsize, truncate=truncate):
                    sampler = BatchSampler(
                        SequentialSampler.from_dataset(self.dset),
                        batchsize, truncate)

                    sampler_iter = iter(sampler)
                    for i in range(len(sampler)):
                        batch = next(sampler_iter)

                        if i < len(sampler) - 1:
                            self.assertEquals(len(batch), batchsize)

                        else:
                            if truncate:
                                self.assertLessEqual(len(batch), batchsize)

    @unittest.skipUnless(check_for_no_backend(),
                         "Test should only be executed "
                         "if no backend is installed/specified")
    def test_sequential(self):
        prev_index = None
        sampler = SequentialSampler.from_dataset(self.dset)

        for idx in sampler:
            if prev_index is not None:
                self.assertEquals(idx, prev_index + 1)

            prev_index = idx

    @unittest.skipUnless(check_for_no_backend(),
                         "Test should only be executed "
                         "if no backend is installed/specified")
    def test_random_replacement(self):

        sampler = RandomSamplerWithReplacement.from_dataset(self.dset)
        samples = []

        self.assertEquals(len(sampler), len(self.dset))

        for idx in sampler:
            self.assertIn(idx, np.arange(len(self.dset)))
            samples.append(idx)

        # check if all samples are only sampled once (extremly unlikely)
        self.assertFalse((np.bincount(samples) == 1).all())

    @unittest.skipUnless(check_for_no_backend(),
                         "Test should only be executed "
                         "if no backend is installed/specified")
    def test_random_no_replacement(self):

        sampler = RandomSamplerNoReplacement.from_dataset(self.dset)
        samples = []

        self.assertEquals(len(sampler), len(self.dset))

        for idx in sampler:
            self.assertIn(idx, np.arange(len(self.dset)))
            samples.append(idx)

        # check if all samples are only sampled once
        self.assertTrue((np.bincount(samples) == 1).all())

    @unittest.skipUnless(check_for_no_backend(),
                         "Test should only be executed "
                         "if no backend is installed/specified")
    def test_prevalence_sampler(self):

        sampler = PrevalenceRandomSampler.from_dataset(self.dset)
        sample_classes = []

        for idx in sampler:
            self.assertIn(idx, np.arange(len(self.dset)))

            sample_classes.append(self.dset[idx]["label"])

        num_samples_per_class = np.bincount(sample_classes)

        self.assertTrue(
            (num_samples_per_class.min() - num_samples_per_class.max()) <= 1)

    @unittest.skipUnless(check_for_no_backend(),
                         "Test should only be executed "
                         "if no backend is installed/specified"
                         )
    def test_abstract_sampler_iter(self):
        sampler = AbstractSampler.from_dataset(self.dset)

        with self.assertRaises(NotImplementedError):
            iter(sampler)


if __name__ == '__main__':
    unittest.main()
