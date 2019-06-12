from collections import OrderedDict

from numpy import concatenate
from numpy.random import choice, shuffle

from .abstract_sampler import AbstractSampler
from ..dataset import AbstractDataset


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
        super().__init__(indices)
        self._indices = list(range(len(indices)))

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
        n_indices = self._check_batchsize(n_indices)

        indices = choice(self._indices,
                         size=n_indices)

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
        super().__init__(indices)

        self._num_samples = 0
        _indices = {}
        for idx, class_idx in enumerate(indices):
            self._num_samples += 1
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
        n_indices = self._check_batchsize(n_indices)

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
        if self._shuffle:
            shuffle(_samples)

        return _samples

    def __len__(self):
        return self._num_samples


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
        super().__init__(indices)

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

        self._num_samples = length

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

    def _check_batchsize(self, n_indices):
        """
        Checks if batchsize is valid for all classes

        Parameters
        ----------
        n_indices : int
            the number of samples to return

        Returns
        -------
        dict
            number of samples per class to return

        """
        n_indices = super()._check_batchsize(n_indices)

        samples_per_class = n_indices // self._n_classes
        remaining = n_indices % self._n_classes

        samples = {}

        try:

            # sample same number of sample for each class
            for key, idx_list in self._indices.items():
                if self._global_idxs[key] >= len(idx_list):
                    raise StopIteration

                # truncate if necessary
                samples[key] = min(
                    samples_per_class,
                    len(self._indices[key]) - self._global_idxs[key])

                self._global_idxs[key] += samples[key]

            # fill up starting with largest class
            while remaining:
                for key, idx_list in self._indices.items():
                    samples[key] += 1
                    remaining -= 1

        except StopIteration as e:
            # set all global indices to 0
            for key in self._global_idxs.keys():
                self._global_idxs[key] = 0

            raise e

        finally:
            return samples

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
        n_indices = self._check_batchsize(n_indices)

        samples = []

        for key, _n_indices in n_indices.items():
            samples.append(choice(self._indices[key], size=_n_indices))

        samples = concatenate(samples)

        if self._shuffle:
            shuffle(samples)

        return samples

    def __len__(self):
        return self._num_samples
