from delira.data_loading.sampler.abstract import AbstractSampler


class BatchSampler(object):
    """
    A Sampler-Wrapper combining the single indices sampled by a sampler to
    batches of a given size
    """

    def __init__(self, sampler: AbstractSampler, batch_size, drop_last=False):
        """

        Parameters
        ----------
        sampler : :class:`AbstractSampler`
            the actual sampler producing single-sized samples
        batch_size : int
            the size of each batch
        drop_last : bool
            whether or not to discard the last (possibly smaller) batch
        """
        self._sampler = sampler
        self._batchsize = batch_size
        self._drop_last = drop_last

    def __iter__(self):
        """
        Iterator holding lists of sample-indices. Each list contains indices
        for a single batch

        Yields
        ------
        list
            a list containing the sample indices of the current batch

        """
        batch_idxs = []

        for idx in self._sampler:
            batch_idxs.append(idx)

            if len(batch_idxs) == self._batchsize:
                yield batch_idxs

                batch_idxs = []

        if not self._drop_last and batch_idxs:
            yield batch_idxs

    def __len__(self):
        """
        Defines the class length

        Returns
        -------
        int
            number of samples

        """
        num_batches = len(self._sampler) // self._batchsize

        if not self._drop_last:
            num_batches += int(bool(len(self._sampler) % self._batchsize))

        return num_batches
