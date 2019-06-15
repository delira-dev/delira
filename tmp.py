import numpy as np
from delira.data_loading import AbstractDataset, BaseDataManager


class DummyDataset(AbstractDataset):
    def __init__(self, length):
        super().__init__(None, None)
        self._length = length

    def __getitem__(self, item):
        return {"data": np.random.rand(1, 2), "label": np.random.rand(1, 1)}

    def __len__(self):
        return self._length

    def get_sample_from_index(self, index):
        return self[index]


if __name__ == '__main__':
    LEN_DSET = 500
    BATCH_SIZE = 4
    NUM_PROCESS = 4
    NUM_BATCHES = LEN_DSET // BATCH_SIZE + int(bool(LEN_DSET % BATCH_SIZE))

    dset = DummyDataset(LEN_DSET)

    mgr = BaseDataManager(dset, BATCH_SIZE, NUM_PROCESS, transforms=None)
    batchgen = mgr.get_batchgen()

    assert mgr.n_batches == NUM_BATCHES
    assert batchgen.num_batches == NUM_BATCHES
