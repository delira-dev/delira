from abc import abstractmethod

from ..dataset import AbstractDataset


class AbstractSampler(object):
    """
    Class to define an abstract Sampling API

    """

    def __init__(self, indices=None):
        self._num_samples = len(indices)
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
        :class:`AbstractSampler`
            The initialzed sampler

        """
        indices = list(range(len(dataset)))
        return cls(indices, **kwargs)

    def _check_batchsize(self, n_indices):
        """
        Checks if the batchsize is valid (and truncates batches if necessary).
        Will also raise StopIteration if enough batches sampled

        Parameters
        ----------
        n_indices : int
            number of indices to sample

        Returns
        -------
        int
            number of indices to sample (truncated if necessary)

        Raises
        ------
        StopIteration
            if enough batches sampled

        """

        if self._global_index >= self._num_samples:
            self._global_index = 0
            raise StopIteration

        else:
            # truncate batch if necessary
            if n_indices + self._global_index > self._num_samples:
                n_indices = self._num_samples - self._global_index

        self._global_index += n_indices
        return n_indices

    @abstractmethod
    def _get_indices(self, n_indices):
        """
        Function to return a specific number of indices.
        Implements the actual sampling strategy.

        Parameters
        ----------
        n_indices : int
            Number of indices to return

        Returns
        -------
        list
            List with sampled indices

        """
        raise NotImplementedError

    def __call__(self, n_indices):
        """
        Function to call the `get_indices` method of the sampler

        Parameters
        ----------
        n_indices : int
            Number of indices to return

        Returns
        -------
        list
            List with sampled indices

        """
        return self._get_indices(n_indices)

    @abstractmethod
    def __len__(self):
        raise NotImplementedError
