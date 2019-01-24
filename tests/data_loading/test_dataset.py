import numpy as np

from delira.data_loading import TorchvisionClassificationDataset, \
    ConcatDataset, BaseCacheDataset

def test_data_subset_concat():
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
    assert len(concat_dataset) == (len(dset_a) + len(dset_b))

    assert concat_dataset[0]

    # test slicing:
    half_len_a = len(dset_a) // 2
    half_len_b = len(dset_b) // 2
    assert len(dset_a.get_subset(range(half_len_a))) == half_len_a
    assert len(dset_b.get_subset(range(half_len_b))) == half_len_b

    sliced_concat_set = concat_dataset.get_subset(
        range(half_len_a + half_len_b))
    assert len(sliced_concat_set) == (half_len_a + half_len_b)

    # check if entries are valid
    assert sliced_concat_set[0]


if __name__ == "__main__":
    test_data_subset_concat()
