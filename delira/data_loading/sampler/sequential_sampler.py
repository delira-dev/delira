from random import shuffle
from collections import OrderedDict
from .abstract_sampler import AbstractSampler
from ..dataset import AbstractDataset


class SequentialSampler(AbstractSampler):
    """
    Implements Sequential Sampling from whole Dataset
    """
    def __init__(self, indices):
        """

        Parameters
        ----------
        indices : list
            list of classes each sample belongs to. List index corresponds to
            data index and the value at a certain index indicates the
            corresponding class
        """
        super().__init__()

        self._indices = list(range(len(indices)))
        self._global_index = 0

    def _get_indices(self, n_indices):
        """
        Actual Sampling

        Parameters
        ----------
        n_indices : int
            number of indices to return

        Raises
        ------
        StopIteration : If end of dataset reached

        Returns
        -------
        list
            list of sampled indices
        """
        if self._global_index >= len(self._indices):
            self._global_index = 0
            raise StopIteration

        new_global_idx = self._global_index + n_indices

        # If we reach end, make batch smaller
        if new_global_idx >= len(self._indices):
            new_global_idx = len(self._indices)

        idxs = list(range(self._global_index, new_global_idx))
        self._global_index = new_global_idx

        return idxs

    def __len__(self):
        return len(self._indices)


class PrevalenceSequentialSampler(AbstractSampler):
    """
    Implements Per-Class Sequential sampling and ensures same
    number of samples per batch for each class; If out of samples for one
    class: restart at first sample

    """

    def __init__(self, indices, shuffle_batch=True):
        """

        Parameters
        ----------
        indices : list
            list of classes each sample belongs to. List index corresponds to
            data index and the value at a certain index indicates the
             corresponding class
        shuffle_batch : bool
            if False: indices per class will be returned in a sequential way
            (first: indices belonging to class 1, second: indices belonging
            to class 2 etc.)
            if True: indices will be sampled in a sequential way per class and
            sampled indices will be shuffled
        """
        super().__init__()

        _indices = {}
        _global_idxs = {}
        for idx, class_idx in enumerate(indices):
            class_idx = int(class_idx)
            if class_idx in _indices.keys():
                _indices[class_idx].append(idx)
            else:
                _indices[class_idx] = [idx]
                _global_idxs[class_idx] = 0

        # sort classes after descending number of elements
        ordered_dict = OrderedDict()

        length = 0
        for k in sorted(_indices, key=lambda k: len(_indices[k]),
                        reverse=True):
            ordered_dict[k] = _indices[k]
            length += len(_indices[k])

        self._length = length

        self._indices = ordered_dict
        self._n_classes = len(_indices.keys())
        self._global_idxs = _global_idxs
        self._global_total_idx = 0

        self._shuffle = shuffle_batch

    @classmethod
    def from_dataset(cls, dataset: AbstractDataset, **kwargs):
        indices = range(len(dataset))
        labels = [dataset[idx]['label'] for idx in indices]
        return cls(labels, **kwargs)

    def _get_indices(self, n_indices):
        """
        Actual Sampling

        Parameters
        ----------
        n_indices : int
            number of indices to return

        Raises
        ------
        StopIteration : If end of class indices is reached

        Returns
        -------
        list
            list of sampled indices
        """
        samples_per_class = int(n_indices / self._n_classes)

        if self._global_total_idx >= len(self._indices):
            self._global_total_idx = 0
            raise StopIteration

        _samples = []

        for key, idx_list in self._indices.items():
            if self._global_idxs[key] >= len(idx_list):
                self._global_idxs[key] = 0

            new_global_idx = self._global_idxs[key] + samples_per_class

            if new_global_idx >= len(idx_list):
                new_global_idx = len(idx_list)

            _samples += list(range(self._global_idxs[key], new_global_idx))
            self._global_idxs[key] = new_global_idx

        for key, idx_list in self._indices.items():
            if len(_samples) >= n_indices:
                break

            if self._global_idxs[key] >= len(idx_list):
                self._global_idxs[key] = 0

            new_global_idx = self._global_idxs[key] + 1

            _samples += list(range(self._global_idxs[key], new_global_idx))
            self._global_idxs[key] = new_global_idx

        self._global_total_idx += n_indices

        if self._shuffle:
            shuffle(_samples)

        return _samples

    def __len__(self):
        return self._length


class StoppingPrevalenceSequentialSampler(AbstractSampler):
    """
    Implements Per-Class Sequential sampling and ensures same
    number of samples per batch for each class; Stops if all samples of
    first class have been sampled

    """
    def __init__(self, indices, shuffle_batch=True):
        """

        Parameters
        ----------
        indices : list
            list of classes each sample belongs to. List index corresponds to
            data index and the value at a certain index indicates the
             corresponding class
        shuffle_batch : bool
            if False: indices per class will be returned in a sequential way
            (first: indices belonging to class 1, second: indices belonging
            to class 2 etc.)
            if True: indices will be sampled in a sequential way per class and
            sampled indices will be shuffled
        """
        super().__init__()

        _indices = {}
        _global_idxs = {}
        for idx, class_idx in enumerate(indices):
            class_idx = int(class_idx)
            if class_idx in _indices.keys():
                _indices[class_idx].append(idx)
            else:
                _indices[class_idx] = [idx]
                _global_idxs[class_idx] = 0

        # sort classes after descending number of elements
        ordered_dict = OrderedDict()

        length = float('inf')
        for k in sorted(_indices, key=lambda k: len(_indices[k]),
                        reverse=True):
            ordered_dict[k] = _indices[k]
            length = min(length, len(_indices[k]))

        self._length = length

        self._indices = ordered_dict
        self._n_classes = len(_indices.keys())
        self._global_idxs = _global_idxs

        self._shuffle = shuffle_batch

    @classmethod
    def from_dataset(cls, dataset: AbstractDataset):
        indices = range(len(dataset))
        labels = [dataset[idx]['label'] for idx in indices]
        return cls(labels)

    def _get_indices(self, n_indices):
        """
        Actual Sampling

        Parameters
        ----------
        n_indices : int
            number of indices to return

        Raises
        ------
        StopIteration : If end of class indices is reached for one class

        Returns
        -------
        list
            list of sampled indices

        """
        samples_per_class = int(n_indices/self._n_classes)
        _samples = []

        for key, idx_list in self._indices.items():
            if self._global_idxs[key] >= len(idx_list):
                self._global_idxs[key] = 0
                raise StopIteration

            new_global_idx = self._global_idxs[key] + samples_per_class

            if new_global_idx >= len(idx_list):
                new_global_idx = len(idx_list)

            _samples += list(range(self._global_idxs[key], new_global_idx))
            self._global_idxs[key] = new_global_idx

        for key, idx_list in self._indices.items():
            if len(_samples) >= n_indices:
                break

            if self._global_idxs[key] >= len(idx_list):
                self._global_idxs[key] = 0

            new_global_idx = self._global_idxs[key] + 1

            _samples += list(range(self._global_idxs[key], new_global_idx))
            self._global_idxs[key] = new_global_idx

        if self._shuffle:
            shuffle(_samples)

        return _samples

    def __len__(self):
        return self._length
