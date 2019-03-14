from ..dataset import AbstractDataset
from .abstract_sampler import AbstractSampler

from numpy.random import choice
import numpy as np


class WeightedRandomSampler(AbstractSampler):
    """
    Implements Weighted Random Sampling

    """
    def __init__(self, indices, weights=None):
        """

        Parameters
        ----------
        indices : list
            list of classes each sample belongs to. List index corresponds to
            data index and the value at a certain index indicates the
             corresponding class
        weights : Any or None
            sampling weights; for more details see numpy.random.choice (parameter ``p``

        """
        super().__init__()

        self._indices = list(range(len(indices)))
        self._weights = weights
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
            The initialzed sampler

        """
        labels = [d['label'] for d in dataset]
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
        ValueError
            if weights or cum_weights don't match the population

        """
        if self._global_index >= len(self._indices):
            self._global_index = 0
            raise StopIteration

        new_global_idx = self._global_index + n_indices

        # If we reach end, make batch smaller
        if new_global_idx >= len(self._indices):
            new_global_idx = len(self._indices)

        samples = choice(self._indices,
                         size=new_global_idx - self._global_index,
                         p=self._weights)
        
        self._global_index = new_global_idx
        return samples

    def __len__(self):
        return len(self._indices)


class WeightedPrevalenceRandomSampler(WeightedRandomSampler):
    def __init__(self, indices):
        """
        Implements random Per-Class Sampling and ensures uniform sampling
        of all classes

        Parameters
        ----------
        indices : array-like
            list of classes each sample belongs to. List index corresponds to
            data index and the value at a certain index indicates the
             corresponding class
        """
        weights = np.array(indices).astype(np.float)
        classes, classes_count = np.unique(indices, return_counts=True)

        # compute probabilities
        target_prob = 1/classes.shape[0]

        # generate weight matrix
        for i, c in enumerate(classes):
            weights[weights == c] = (target_prob/classes_count[i])

        super().__init__(indices, weights=weights)
