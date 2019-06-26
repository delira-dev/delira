from delira.data_loading.sampler.abstract_sampler import AbstractSampler


class BatchSampler(object):
    def __init__(self, sampler: AbstractSampler, batch_size, truncate=False):
        self._sampler = sampler
        self._batchsize = batch_size
        self._truncate = truncate

    def __iter__(self):
        batch_idxs = []

        for idx in self._sampler:
            batch_idxs.append(idx)

            if len(batch_idxs) == self._batchsize:
                yield batch_idxs

                batch_idxs = []

        if not self._truncate and batch_idxs:
            yield batch_idxs

    def __len__(self):
        num_batches = len(self._sampler) // self._batchsize

        if not self._truncate:
            num_batches += int(bool(len(self._sampler) % self._batchsize))

        return num_batches
