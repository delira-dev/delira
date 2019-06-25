from delira.data_loading.sampler.abstract_sampler import AbstractSampler


class LambdaSampler(AbstractSampler):
    """
    Implements Arbitrary Sampling methods specified by a function which takes
    the index_list and returns a single index
    """

    def __init__(self, indices, sampling_fn):
        """
        Parameters
        ----------
        indices : list
            list of classes each sample belongs to. List index corresponds to
            data index and the value at a certain index indicates the
             corresponding class
        sampling_fn : function
            Actual sampling implementation; must accept an index-list
            and return a single index
        """
        super().__init__(indices)
        self._indices = list(range(len(indices)))

        self._sampling_fn = sampling_fn

    def _get_next_index(self):
        """
        Actual Sampling
        Returns
        -------
        int
            the next sample
        """
        return self._sampling_fn(self._indices)
