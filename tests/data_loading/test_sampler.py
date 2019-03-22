from delira.data_loading.sampler import LambdaSampler, \
    PrevalenceRandomSampler, \
    PrevalenceSequentialSampler, \
    RandomSampler, \
    SequentialSampler, \
    StoppingPrevalenceRandomSampler, \
    StoppingPrevalenceSequentialSampler, \
    WeightedRandomSampler


import numpy as np
from . import DummyDataset


import unittest

class SamplerTest(unittest.TestCase):

    def test_lambda_sampler(self):
        np.random.seed(1)
        dset = DummyDataset(600, [0.5, 0.3, 0.2])

        def sampling_fn_a(index_list, n_indices):
            return index_list[:n_indices]


        def sampling_fn_b(index_list, n_indices):
            return index_list[-n_indices:]

        sampler_a = LambdaSampler(list(range(len(dset))), sampling_fn_a)
        sampler_b = LambdaSampler(list(range(len(dset))), sampling_fn_b)

        self.assertEqual(sampler_a(15), list(range(15)))
        self.assertEqual(sampler_b(15), list(range(len(dset) - 15, len(dset))))

    def test_prevalence_random_sampler(self):
        np.random.seed(1)
        dset = DummyDataset(600, [0.5, 0.3, 0.2])

        sampler = PrevalenceRandomSampler.from_dataset(dset)

        for batch_len in [1, 2, 3]:
            with self.subTest(batch_len=batch_len):

                equal_batch = sampler(batch_len)

                seen_labels = []
                for idx in equal_batch:
                    curr_label = dset[idx]["label"]

                    self.assertNotIn(curr_label, seen_labels)
                    seen_labels.append(curr_label)

        self.assertEqual(len(sampler(5)), 5)

    def test_prevalence_sequential_sampler(self):
        np.random.seed(1)
        dset = DummyDataset(600, [0.5, 0.3, 0.2])

        sampler = PrevalenceSequentialSampler.from_dataset(dset)

        # ToDo add test considering actual sampling strategy
        self.assertEqual(len(sampler(5)), 5)

    def test_random_sampler(self):
        np.random.seed(1)
        dset = DummyDataset(600, [0.5, 0.3, 0.2])

        sampler = RandomSampler.from_dataset(dset)

        self.assertEqual(len(sampler(250)), 250)

        # checks if labels are all the same (should not happen if random sampled)
        self.assertGreater(
            len(set([dset[_idx]["label"] for _idx in sampler(301)])), 1)

    def test_sequential_sampler(self):
        np.random.seed(1)
        dset = DummyDataset(600, [0.5, 0.3, 0.2])

        sampler = SequentialSampler.from_dataset(dset)

        # if sequentially sampled, the first 300 items should have label 0 -> 1
        # unique element
        self.assertEqual(len(set([dset[_idx]["label"]
                                  for _idx in sampler(100)])), 1)
        self.assertEqual(len(sampler(100)), 100)

        # next 100 elements also same label -> next 201 elements: two different
        # labels
        self.assertEqual(len(set([dset[_idx]["label"]
                                  for _idx in sampler(101)])), 2)

    def test_stopping_prevalence_random_sampler(self):
        np.random.seed(1)
        dset = DummyDataset(600, [0.5, 0.3, 0.2])

        sampler = StoppingPrevalenceRandomSampler.from_dataset(dset)

        with self.assertRaises(StopIteration):
            for i in range(121):
                sample = sampler(3)
                self.assertEqual(
                    len(set(dset[_idx]["label"] for _idx in sample)), 3)

    def test_stopping_prevalence_sequential_sampler(self):
        np.random.seed(1)
        dset = DummyDataset(600, [0.5, 0.3, 0.2])

        sampler = StoppingPrevalenceRandomSampler.from_dataset(dset)

        with self.assertRaises(StopIteration):
            for i in range(121):
                sample = sampler(3)
                self.assertEqual(
                    len(set([dset[_idx]["label"] for _idx in sample])), 3)


def test_weighted_sampler():
    np.random.seed(1)
    dset = DummyDataset(600, [0.5, 0.3, 0.2])

    sampler = WeightedRandomSampler.from_dataset(dset)

    assert len(sampler(250)) == 250

    # checks if labels are all the same (should not happen if random sampled)
    assert len(set([dset[_idx]["label"] for _idx in sampler(301)])) > 1


def test_weighted_prevalence_sampler():
    np.random.seed(1)
    dset = DummyDataset(2000, [0.5, 0.3, 0.2])

    sampler = WeightedPrevalenceRandomSampler.from_dataset(dset)

    assert len(sampler(250)) == 250

    # checks if labels are all the same (should not happen if random sampled)
    n_draw = 1000
    label_list = [dset[_idx]["label"] for _idx in sampler(n_draw)]
    assert len(set(label_list)) > 1
    assert abs(label_list.count(0)/n_draw - (1 / 3)) < 0.1
    assert abs(label_list.count(1)/n_draw - (1 / 3)) < 0.1
    assert abs(label_list.count(2)/n_draw - (1 / 3)) < 0.1


if __name__ == '__main__':
    unittest.main()
