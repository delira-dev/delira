from collections import OrderedDict
from ..dataset import AbstractDataset
from .abstract_sampler import AbstractSampler

from numpy.random import choice, shuffle
from numpy import concatenate


class RandomSampler(AbstractSampler):
    """
    Implements Random Sampling from whole Dataset
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
        n_indices: int
            number of indices to return

        Returns
        -------
        list
            list of sampled indices

        Raises
        ------
        StopIteration
            If maximal number of samples is reached

        """
        if self._global_index >= len(self._indices):
            self._global_index = 0
            raise StopIteration

        new_global_idx = self._global_index + n_indices

        # If we reach end, make batch smaller
        if new_global_idx >= len(self._indices):
            new_global_idx = len(self._indices)

        indices = choice(self._indices,
                         size=new_global_idx - self._global_index)
        # indices = choices(self._indices, k=new_global_idx - self._global_index)
        self._global_index = new_global_idx
        return indices

    def __len__(self):
        return len(self._indices)


class PrevalenceRandomSampler(AbstractSampler):
    """
    Implements random Per-Class Sampling and ensures same
    number of samplers per batch for each class

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
            to class 2 etc.) while indices for each class are sampled in a
            random way
            if True: indices will be sampled in a random way per class and
            sampled indices will be shuffled

        """
        super().__init__()

        self._num_indices = 0
        _indices = {}
        for idx, class_idx in enumerate(indices):
            self._num_indices += 1
            class_idx = int(class_idx)
            if class_idx in _indices.keys():
                _indices[class_idx].append(idx)
            else:
                _indices[class_idx] = [idx]

        # sort classes after descending number of elements
        ordered_dict = OrderedDict()

        for k in sorted(_indices, key=lambda k: len(_indices[k]),
                        reverse=True):
            ordered_dict[k] = _indices[k]

        self._indices = ordered_dict
        self._n_classes = len(_indices.keys())

        self._shuffle = shuffle_batch
        self._global_index = 0

    @classmethod
    def from_dataset(cls, dataset: AbstractDataset, **kwargs):
        """
        Classmethod to initialize the sampler from a given dataset

        Parameters
        ----------
        dataset : AbstractDataset
            the given dataset

        Returns
        -------
        AbstractSampler
            The initialized sampler

        """
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

        Returns
        -------
        list
            list of sampled indices

        Raises
        ------
        StopIteration
            If maximal number of samples is reached

        """
        if self._global_index >= self._num_indices:
            self._global_index = 0
            raise StopIteration

        samples_per_class = int(n_indices / self._n_classes)

        _samples = []

        for key, idx_list in self._indices.items():
            _samples.append(choice(idx_list, size=samples_per_class))

        # add elements until len(_samples) == n_indices
        # works because less indices are left than n_classes
        # and self._indices is sorted by decreasing number of elements
        for key, idx_list in self._indices.items():
            if len(_samples) >= n_indices:
                break
            _samples.append(choice(idx_list, size=1))

        _samples = concatenate(_samples)
        self._global_index += n_indices
        if self._shuffle:
            shuffle(_samples)

        return _samples

    def __len__(self):
        return self._num_indices


class StoppingPrevalenceRandomSampler(AbstractSampler):
    """
    Implements random Per-Class Sampling and ensures same
    number of samplers per batch for each class; Stops if out of samples for
    smallest class

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
    def from_dataset(cls, dataset: AbstractDataset, **kwargs):
        """
        Classmethod to initialize the sampler from a given dataset

        Parameters
        ----------
        dataset : AbstractDataset
            the given dataset

        Returns
        -------
        AbstractSampler
            The initialized sampler

        """
        indices = range(len(dataset))
        labels = [dataset[idx]['label'] for idx in indices]
        return cls(labels, **kwargs)

    def _get_indices(self, n_indices):
        """
        Actual Sampling

        Parameters
        ----------
        n_indices: int
            number of indices to return

        Raises
        ------
        StopIteration: If end of class indices is reached for one class

        Returns
        -------
            list: list of sampled indices
        """
        samples_per_class = int(n_indices / self._n_classes)
        _samples = []

        for key, idx_list in self._indices.items():
            if self._global_idxs[key] >= len(idx_list):
                self._global_idxs[key] = 0
                raise StopIteration

            new_global_idx = self._global_idxs[key] + samples_per_class

            if new_global_idx >= len(idx_list):
                new_global_idx = len(idx_list)

            _samples.append(choice(idx_list, size=samples_per_class))

            self._global_idxs[key] = new_global_idx

        for key, idx_list in self._indices.items():
            if len(_samples) >= n_indices:
                break

            if self._global_idxs[key] >= len(idx_list):
                self._global_idxs[key] = 0

            new_global_idx = self._global_idxs[key] + 1

            _samples.append(choice(idx_list, size=1))
            self._global_idxs[key] = new_global_idx

        _samples = concatenate(_samples)
        if self._shuffle:
            shuffle(_samples)

        return _samples

    def __len__(self):
        return self._length
