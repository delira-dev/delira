from delira.data_loading import BaseDataLoader, SequentialSampler

import numpy as np 
from . import DummyDataset

def test_data_loader():
    np.random.seed(1)
    dset = DummyDataset(600, [0.5, 0.3, 0.2])
    sampler = SequentialSampler.from_dataset(dset)
    loader = BaseDataLoader(dset, batch_size=16, sampler=sampler)

    assert isinstance(loader.generate_train_batch(), dict)
    for key, val in loader.generate_train_batch().items():
        assert len(val) == 16
    assert "label" in loader.generate_train_batch()
    assert "data" in loader.generate_train_batch()
    assert len(set([_tmp for _tmp in loader.generate_train_batch()["label"]])) == 1

if __name__ == '__main__':
    test_data_loader()