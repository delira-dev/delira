from abc import abstractmethod

from delira.data_loading.dataset import AbstractDataset


class AbstractSampler(object):
    """
    Class to define an abstract Sampling API
    """

    def __init__(self, indices=None):
        self._num_samples = len(indices)

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
    def _get_next_index(self):
        """
        Function to return a single index.
        Implements the actual sampling strategy.
        Returns
        -------
        int
            the next index
        """
        raise NotImplementedError

    def __iter__(self):
        try:
            for i in range(self._num_samples):
                yield self._get_next_index()
        except StopIteration:
            return

    def __len__(self):
        return self._num_samples
