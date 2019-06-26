from numpy.random import choice

from delira.data_loading.sampler.abstract_sampler import AbstractSampler
from delira.data_loading.sampler.per_class_sampler import PerClassSampler, \
    StoppingPerClassSampler


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

    def _get_next_index(self):
        """
        Actual Sampling
        Returns
        -------
        int
            next index
        """

        return choice(self._indices)


class PerClassRandomSampler(PerClassSampler):
    """
    Implements random Per-Class Sampling by returning an index of the first
    class, moving to next class, returning an index for the second class,
    moving on etc.; If out of samples for one class, the class will be skipped
    during sampling
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
        super().__init__(indices, RandomSampler)


class StoppingPerClassRandomSampler(StoppingPerClassSampler):
    """
    Implements random Per-Class Sampling by returning an index of the first
    class, moving to next class, returning an index for the second class,
    moving on etc.; Stops if out of samples for one class
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
        super().__init__(indices, RandomSampler)
