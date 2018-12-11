from .abstract_sampler import AbstractSampler


class LambdaSampler(AbstractSampler):
    """
    Implements Arbitrary Sampling methods specified by a function which takes
    the index_list and the number of indices to return

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
            and the number of indices to return

        """
        super().__init__()
        self._indices = list(range(len(indices)))

        self._sampling_fn = sampling_fn
        self._global_index = 0

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
            Maximum number of indices sampled

        """

        if self._global_index >= len(self._indices):
            self._global_index = 0
            raise StopIteration

        new_global_idx = self._global_index + n_indices

        # If we reach end, make batch smaller
        if new_global_idx >= len(self._indices):
            new_global_idx = len(self._indices)

        samples = self._sampling_fn(self._indices,
                                    new_global_idx - self._global_index)

        self._global_index = new_global_idx
        return samples

    def __len__(self):
        return len(self._indices)
