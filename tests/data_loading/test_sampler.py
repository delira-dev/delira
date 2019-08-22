# ToDo: Add sampler tests

# import unittest
#
# import numpy as np
#
# from delira.data_loading.sampler import LambdaSampler, \
#     PerClassRandomSampler, \
#     PerClassSequentialSampler, \
#     RandomSampler, \
#     StoppingPerClassRandomSampler, \
#     SequentialSampler, \
#     StoppingPrevalenceRandomSampler, \
#     WeightedRandomSampler, \
#     WeightedPrevalenceRandomSampler, \
#     BatchSampler, \
#     StoppingPerClassSequentialSampler
#
# from ..utils import check_for_no_backend
#
# from .utils import DummyDataset
#
#
# class SamplerTest(unittest.TestCase):
#
#     @unittest.skipUnless(check_for_no_backend(),
#                          "Test should be only executed if no "
#                          "backend was installed")
#     def test_lambda_sampler(self):
#         np.random.seed(1)
#         dset = DummyDataset(600, [0.5, 0.3, 0.2])
#
#         class SamplingFnA(object):
#             def __init__(self, index_list):
#                 self._indices = index_list
#                 self._iter = iter(self._indices)
#
#             def __call__(self):
#                 try:
#                     return next(self._iter)
#                 except StopIteration:
#                     self._iter = iter(self._indices)
#                     return self()
#
#         class SamplingFnB(SamplingFnA):
#             def __init__(self, index_list):
#                 index_list = reversed(index_list)
#                 super().__init__(index_list)
#
#         sampler_a = BatchSampler(LambdaSampler(list(range(len(dset))),
#                                                SamplingFnA), 15)
#         sampler_b = BatchSampler(LambdaSampler(list(range(len(dset))),
#                                                SamplingFnB), 15)
#
#         sampler_iter_a = iter(sampler_a)
#         sampler_iter_b = iter(sampler_b)
#         self.assertEqual(next(sampler_iter_a), list(range(15)))
#         self.assertEqual(next(sampler_iter_b),
#                          list(range(len(dset), len(dset) - 15, -1)))
#
#     @unittest.skipUnless(check_for_no_backend(),
#                          "Test should be only executed if no "
#                          "backend was installed")
#     def test_prevalence_random_sampler(self):
#         np.random.seed(1)
#         dset = DummyDataset(600, [0.5, 0.3, 0.2])
#
#         sampler = PerClassRandomSampler.from_dataset(dset)
#
#         for batch_len in [1, 2, 3]:
#             with self.subTest(batch_len=batch_len):
#                 batch_sampler = BatchSampler(sampler, batch_len)
#                 sampler_iter = iter(batch_sampler)
#                 equal_batch = next(sampler_iter)
#
#                 seen_labels = []
#                 for idx in equal_batch:
#                     curr_label = dset[idx]["label"]
#
#                     self.assertNotIn(curr_label, seen_labels)
#                     seen_labels.append(curr_label)
#
#         batch_sampler = BatchSampler(sampler, 5)
#         sampler_iter = iter(batch_sampler)
#         self.assertEqual(len(next(sampler_iter)), 5)
#
#     @unittest.skipUnless(check_for_no_backend(),
#                          "Test should be only executed if no "
#                          "backend was installed")
#     def test_prevalence_sequential_sampler(self):
#         np.random.seed(1)
#         dset = DummyDataset(600, [0.5, 0.3, 0.2])
#
#         sampler = PerClassSequentialSampler.from_dataset(dset)
#
#         # ToDo add test considering actual sampling strategy
#         batch_sampler = BatchSampler(sampler, 5)
#         sampler_iter = iter(batch_sampler)
#         self.assertEqual(len(next(sampler_iter)), 5)
#
#     @unittest.skipUnless(check_for_no_backend(),
#                          "Test should be only executed if no "
#                          "backend was installed")
#     def test_random_sampler(self):
#         np.random.seed(1)
#         dset = DummyDataset(600, [0.5, 0.3, 0.2])
#
#         sampler = RandomSampler.from_dataset(dset)
#
#         batch_sampler = BatchSampler(sampler, 250)
#         sampler_iter = iter(batch_sampler)
#         self.assertEqual(len(next(sampler_iter)), 250)
#
#         # checks if labels are all the same (should not happen if random
#         # sampled)
#         self.assertGreater(
#             len(set([dset[_idx]["label"]
#                      for _idx in next(iter(BatchSampler(sampler, 301)))])), 1)
#
#     @unittest.skipUnless(check_for_no_backend(),
#                          "Test should be only executed if no "
#                          "backend was installed")
#     def test_sequential_sampler(self):
#         np.random.seed(1)
#         dset = DummyDataset(600, [0.5, 0.3, 0.2])
#
#         sampler = SequentialSampler.from_dataset(dset)
#
#         # if sequentially sampled, the first 300 items should have label 0 -> 1
#         # unique element
#         batch_sampler = BatchSampler(sampler, 100)
#         batch_sampler_iter = iter(batch_sampler)
#         self.assertEqual(len(set([dset[_idx]["label"]
#                                   for _idx in next(batch_sampler_iter)])), 1)
#         self.assertEqual(len(next(batch_sampler_iter)), 100)
#
#         # next 100 elements also same label -> next 201 elements: two different
#         # labels
#         self.assertEqual(len(set([dset[_idx]["label"]
#                                   for _idx
#                                   in [*next(batch_sampler_iter),
#                                       *next(batch_sampler_iter)]])), 2)
#
#     @unittest.skipUnless(check_for_no_backend(),
#                          "Test should be only executed if no "
#                          "backend was installed")
#     def test_stopping_prevalence_random_sampler(self):
#         np.random.seed(1)
#         dset = DummyDataset(600, [0.5, 0.3, 0.2])
#
#         sampler = StoppingPerClassRandomSampler.from_dataset(dset)
#         batchsampler = BatchSampler(sampler, 3)
#         batchsampler_iter = iter(batchsampler)
#
#         with self.assertRaises(StopIteration):
#             for i in range(121):
#                 sample = next(batchsampler_iter)
#                 self.assertEqual(
#                     len(set(dset[_idx]["label"] for _idx in sample)), 3)
#
#     def test_stopping_prevalence_sequential_sampler(self):
#         np.random.seed(1)
#         dset = DummyDataset(600, [0.5, 0.3, 0.2])
#
#         sampler = StoppingPerClassSequentialSampler.from_dataset(dset)
#         batchsampler = BatchSampler(sampler, 3)
#         batchsampler_iter = iter(batchsampler)
#
#         with self.assertRaises(StopIteration):
#             for i in range(121):
#                 sample = next(batchsampler_iter)
#                 self.assertEqual(
#                     len(set([dset[_idx]["label"] for _idx in sample])), 3)
#
#     @unittest.skipUnless(check_for_no_backend(),
#                          "Test should be only executed if no "
#                          "backend was installed")
#     def test_weighted_sampler(self):
#         np.random.seed(1)
#         dset = DummyDataset(600, [0.5, 0.3, 0.2])
#
#         sampler = WeightedRandomSampler.from_dataset(dset)
#         batch_sampler = BatchSampler(sampler, 199)
#         batchsampler_iter = iter(batch_sampler)
#
#         assert len(next(batchsampler_iter)) == 199
#
#         # checks if labels are all the same (should not happen if random
#         # sampled)
#         assert len(set([dset[_idx]["label"]
#                         for _idx in [*next(batchsampler_iter),
#                                      *next(batchsampler_iter)]])) > 1
#
#         # checks if labels are all the same (should not happen if random
#         # sampled)
#         assert len(set([dset[_idx]["label"] for _idx in sampler(301)])) > 1
#
#     @unittest.skipUnless(check_for_no_backend(),
#                          "Test should be only executed if no "
#                          "backend was installed")
#     def test_weighted_prevalence_sampler(self):
#         np.random.seed(1)
#         dset = DummyDataset(2000, [0.5, 0.3, 0.2])
#
#         sampler = WeightedPrevalenceRandomSampler.from_dataset(dset)
#         batchsampler = BatchSampler(sampler, 250)
#         sampler_iter = iter(batchsampler)
#
#         assert len(next(sampler_iter)) == 250
#
#         # checks if labels are all the same (should not happen if random
#         # sampled)
#         n_draw = 1000
#         label_list = []
#         for tmp in range(n_draw // 250):
#             label_list += [dset[_idx]["label"] for _idx in next(sampler_iter)]
#
#         assert len(set(label_list)) > 1
#         assert abs(label_list.count(0) / n_draw - (1 / 3)) < 0.1
#         assert abs(label_list.count(1) / n_draw - (1 / 3)) < 0.1
#         assert abs(label_list.count(2) / n_draw - (1 / 3)) < 0.1
#
#
# if __name__ == '__main__':
#     unittest.main()
