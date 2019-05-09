import numpy as np

from delira.data_loading import ConcatDataset, BaseCacheDataset, \
    BaseExtendCacheDataset, BaseLazyDataset, LoadSample, LoadSampleLabel, \
    AbstractDataset
from delira.data_loading.load_utils import norm_zero_mean_unit_std

import unittest


class DataSubsetConcatTest(unittest.TestCase):

    def test_data_subset_concat(self):

        def load_dummy_sample(path, label_load_fct):
            """
            Returns dummy data, independent of path or label_load_fct
            Parameters
            ----------
            path
            label_load_fct
            Returns
            -------
            : dict
                dict with data and label
            """

            return {'data': np.random.rand(1, 256, 256),
                    'label': np.random.randint(2)}

        class DummyCacheDataset(BaseCacheDataset):
            def __init__(self, num: int, label_load_fct, *args, **kwargs):
                """
                Generates random samples with _make_dataset
                Parameters
                ----------
                num : int
                    number of random samples
                args :
                    passed to BaseCacheDataset
                kwargs :
                    passed to BaseCacheDataset

                """
                self.label_load_fct = label_load_fct
                super().__init__(data_path=num, *args, **kwargs)

            def _make_dataset(self, path):
                data = []
                for i in range(path):
                    data.append(self._load_fn(i, self.label_load_fct))
                return data

        dset_a = DummyCacheDataset(500, None, load_fn=load_dummy_sample,
                                   img_extensions=[], gt_extensions=[])
        dset_b = DummyCacheDataset(700, None, load_fn=load_dummy_sample,
                                   img_extensions=[], gt_extensions=[])

        # test concatenating
        concat_dataset = ConcatDataset(dset_a, dset_b)

        self.assertEqual(len(concat_dataset), len(dset_a) + len(dset_b))

        self.assertTrue(concat_dataset[0])


        # test slicing:
        half_len_a = len(dset_a) // 2
        half_len_b = len(dset_b) // 2

        self.assertEqual(len(dset_a.get_subset(range(half_len_a))), half_len_a)
        self.assertEqual(len(dset_b.get_subset(range(half_len_b))), half_len_b)

        sliced_concat_set = concat_dataset.get_subset(
            range(half_len_a + half_len_b))

        self.assertEqual(len(sliced_concat_set), half_len_a + half_len_b)

        # check if entries are valid
        self.assertTrue(sliced_concat_set[0])


def test_cache_dataset():
    def load_mul_sample(path):
        """
        Return a list of random samples
        Parameters
        ----------
        path

        Returns
        -------
        list
            list of samples
        """
        return [load_dummy_sample(None)] * 4

    # test normal cache dataset
    paths = list(range(10))
    dataset = BaseCacheDataset(paths, load_dummy_sample)
    assert len(dataset) == 10
    try:
        a = dataset[0]
        a = dataset[5]
        a = dataset[9]
    except:
        raise AssertionError('Dataset access failed.')

    try:
        j = 0
        for i in dataset:
            assert 'data' in i
            assert 'label' in i
            j += 1
        assert j == len(dataset)
    except:
        raise AssertionError('Dataset iteration failed.')

    # test extend cache dataset
    dataset = BaseExtendCacheDataset(paths, load_mul_sample)
    assert len(dataset) == 40
    try:
        a = dataset[0]
        a = dataset[20]
        a = dataset[39]
    except:
        raise AssertionError('Dataset access failed.')

    try:
        j = 0
        for i in dataset:
            assert 'data' in i
            assert 'label' in i
            j += 1
        assert j == len(dataset)
    except:
        raise AssertionError('Dataset iteration failed.')


def test_lazy_dataset():
    # test lazy dataset
    paths = list(range(10))
    dataset = BaseLazyDataset(paths, load_dummy_sample)
    assert len(dataset) == 10
    try:
        a = dataset[0]
        a = dataset[5]
        a = dataset[9]
    except:
        raise AssertionError('Dataset access failed.')

    try:
        j = 0
        for i in dataset:
            assert 'data' in i
            assert 'label' in i
            j += 1
        assert j == len(dataset)
    except:
        raise AssertionError('Dataset iteration failed.')


def test_load_sample():
    def load_dummy_label(path):
        return {'label': 42}

    def load_dummy_data(path):
        return np.random.rand(1, 256, 256) * np.random.randint(2, 20) + \
               np.random.randint(20)

    # check loading of a single sample
    sample_fn = LoadSample({'data': ['data', 'data', 'data'], 'seg': ['data'],
                            'data2': ['data', 'data', 'data']},
                           load_dummy_data,
                           dtype={'seg': 'uint8'},
                           normalize=['data2'])
    sample = sample_fn('load')
    assert not np.isclose(np.mean(sample['data']), 0)
    assert not np.isclose(np.mean(sample['seg']), 0)
    assert sample['seg'].dtype == 'uint8'
    assert np.isclose(sample['data2'].max(), 1)
    assert np.isclose(sample['data2'].min(), -1)

    # check different normalization function
    sample_fn = LoadSample({'data': ['data', 'data', 'data']},
                           load_dummy_data,
                           normalize=['data'],
                           norm_fn=norm_zero_mean_unit_std)
    sample = sample_fn('load')
    assert np.isclose(np.mean(sample['data']), 0)
    assert np.isclose(np.std(sample['data']), 1)

    # check label and loading of single sample
    sample_fn = LoadSampleLabel(
        {'data': ['data', 'data', 'data'], 'seg': ['data'],
         'data2': ['data', 'data', 'data']}, load_dummy_data,
        'label', load_dummy_label,
        sample_kwargs={'dtype': {'seg': 'uint8'}, 'normalize': ['data2']})
    sample = sample_fn('load')
    assert not np.isclose(np.mean(sample['data']), 0)
    assert not np.isclose(np.mean(sample['seg']), 0)
    assert sample['seg'].dtype == 'uint8'
    assert np.isclose(sample['data2'].max(), 1)
    assert np.isclose(sample['data2'].min(), -1)
    assert sample['label'] == 42


if __name__ == "__main__":
    unittest.main()

