from abc import abstractmethod
from ..dataset import AbstractDataset


class AbstractSampler(object):
    """
    Class to define an abstract Sampling API

    """
    def __init__(self, indices=None):
        pass

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
