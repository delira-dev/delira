from delira.data_loading.sampler.abstract import AbstractSampler


class SequentialSampler(AbstractSampler):
    """
    Class to implement sequential sampling
    """

    def __iter__(self):
        """
        Creates an iterator returning sequential samples

        Returns
        -------
        Iterator
            iterator returning samples in a sequential manner
        """
        return iter(range(len(self._indices)))
