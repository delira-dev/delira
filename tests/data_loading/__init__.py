from delira.data_loading import AbstractDataset
import numpy as np
import math


class DummyDataset(AbstractDataset):
    def __init__(self, length=600, class_weights=[0.5, 0.3, 0.2]):
        super().__init__(None, None)

        assert math.isclose(sum(class_weights), 1)

        self._data = [np.random.rand(1, 28, 28) for i in range(length)]
        _labels = []
        for idx, weight in enumerate(class_weights):
            _labels += [idx] * int(length*weight)

        self._labels = _labels

    def __getitem__(self, index):
        return {"data": self._data[index], "label": self._labels[index]}

    def __len__(self):
        return len(self._data)