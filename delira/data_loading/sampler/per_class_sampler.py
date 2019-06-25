from delira.data_loading.sampler.abstract_sampler import AbstractSampler
from collections import OrderedDict
from delira.data_loading.dataset import AbstractDataset


class PerClassSampler(AbstractSampler):
    def __init__(self, indices, sampling_class, **kwargs):
        super().__init__(indices)

        self._num_samples = 0
        _indices = {}
        for idx, class_idx in enumerate(indices):
            self._num_samples += 1
            class_idx = int(class_idx)
            if class_idx in _indices.keys():
                _indices[class_idx].append(idx)
            else:
                _indices[class_idx] = [idx]

        # sort classes after descending number of elements
        ordered_dict = OrderedDict()
        self._skip_cls = {}

        for k in sorted(_indices, key=lambda k: len(_indices[k]),
                        reverse=True):
            ordered_dict[k] = sampling_class(_indices[k], **kwargs)
            self._skip_cls[k] = False

        self._samples = ordered_dict
        self._class_iter = iter(self._samples.keys())

    def _get_next_index(self):
        try:
            curr_class = next(self._class_iter)
        except StopIteration:
            self._class_iter = iter(self._samples.keys())
            curr_class = next(self._class_iter)

        if not self._skip_cls[curr_class]:
            return next(self._samples[curr_class])

        return self._get_next_index()

    def __iter__(self):
        if all(self._samples.values()):
            return
        yield self._get_next_index()

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
            The initialized sampler
        """
        indices = range(len(dataset))
        labels = [dataset[idx]['label'] for idx in indices]
        return cls(labels, **kwargs)


class StoppingPerClassSampler(PerClassSampler):
    def _get_next_index(self):
        try:
            curr_class = next(self._class_iter)
        except StopIteration:
            raise GeneratorExit

        return next(self._samples[curr_class])
