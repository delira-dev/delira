from delira.data_loading.sampler.abstract import AbstractSampler
from delira.data_loading.dataset import AbstractDataset
import numpy as np


class WeightedRandomSampler(AbstractSampler):
    """
    Class implementing Weighted Random Sampling
    """

    def __init__(self, weights, num_samples=None):
        """

        Parameters
        ----------
        weights : list
            per-sample weights
        num_samples : int
            number of samples to provide. If not specified this defaults to
            the amount of values given in :param:`num_samples´
        """
        super().__init__(weights)

        if num_samples is None:
            num_samples = len(weights)

        self._num_samples = num_samples

    def __iter__(self):
        """
        Defines the actual weighted random sampling

        Returns
        -------
        Iterator
            iterator producing random samples
        """
        return iter(np.random.multinomial(self._num_samples,
                                          self._indices, size=1))

    def __len__(self):
        """
        Defines the length of the sampler

        Returns
        -------
        int
            the number of samples
        """
        return self._num_samples


class PrevalenceRandomSampler(WeightedRandomSampler):
    """
    Class implementing prevalence weighted sampling
    """

    def __init__(self, indices):
        """

        Parameters
        ----------
        indices : list
            list of class indices to calculate a weighting from
        """
        class_weights = 1 / np.bincount(indices)

        new_weights = [class_weights[_index] for _index in indices]
        super().__init__(new_weights, num_samples=len(indices))

    @classmethod
    def from_dataset(cls, dset: AbstractDataset, key="label", **kwargs):
        """
        CLass function to create an instance of this sampler by giving it a
        dataset

        Parameters
        ----------
        dset : :class:`AbstractDataset`
            the dataset to create weightings from
        key : str
            the key holding the class index for each sample
        **kwargs :
            Additional keyword arguments

        """
        return cls([_sample[key] for _sample in dset], **kwargs)
