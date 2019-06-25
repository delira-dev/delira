import numpy as np
from numpy.random import choice

from delira.data_loading.sampler.abstract_sampler import AbstractSampler
from delira.data_loading.dataset import AbstractDataset


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
            sampling weights; for more details see numpy.random.choice
            (parameter ``p``)
        """
        super().__init__(indices)

        self._indices = list(range(len(indices)))
        self._weights = weights
        self._global_index = 0

    def _get_next_index(self):
        return choice(self._indices, p=self._weights)

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
        target_prob = 1 / classes.shape[0]

        # generate weight matrix
        for i, c in enumerate(classes):
            weights[weights == c] = (target_prob / classes_count[i])

        super().__init__(indices, weights=weights)
