from delira.data_loading.sampler.abstract_sampler import AbstractSampler
from delira.data_loading.sampler.per_class_sampler import PerClassSampler, \
    StoppingPerClassSampler


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
        super().__init__(indices)

        self._indices = iter(range(len(indices)))

    def _get_next_index(self):
        """
        Actual Sampling
        Raises
        ------
        StopIteration : If end of dataset reached
        Returns
        -------
        int
            the next index
        """
        return next(self._indices)

    def __len__(self):
        return self._num_samples


class PerClassSequentialSampler(PerClassSampler):
    """
    Implements Per-Class Sequential sampling by returning an index of the first
    class, moving to next class, returning an index for the second class,
    moving on etc.
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
        super().__init__(indices, SequentialSampler)


class StoppingPerClassSequentialSampler(StoppingPerClassSampler):
    """
    Implements Per-Class Sequential sampling and ensures same
    number of samples per batch for each class; Stops if all samples of
    first class have been sampled
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
        super().__init__(indices, SequentialSampler)
