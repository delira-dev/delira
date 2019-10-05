from delira.data_loading.dataset import AbstractDataset


class AbstractSampler(object):
    """
    Abstract Class defining a sampler interface
    """

    def __init__(self, indices):
        """

        Parameters
        ----------
        indices : list
            the indices containing the classes to sample from
        """
        self._indices = indices

    def __iter__(self):
        """
        Returns an iterator, must be overwritten in subclasses

        Raises
        ------
        NotImplementedError
            if not overwritten in subclass

        """
        raise NotImplementedError

    def __len__(self):
        """
        Defines the class length

        Returns
        -------
        int
            the number of samples

        """
        return len(self._indices)

    @classmethod
    def from_dataset(cls, dset: AbstractDataset, **kwargs):
        """
        Class Method to create a sampler from a given dataset

        Parameters
        ----------
        dset : :class:`AbstractDataset`
            the dataset to create the sampler from
        **kwargs :
            additional keyword arguments

        """
        if hasattr(dset, "__len__"):
            length = len(dset)
        else:
            length = len([tmp for tmp in dset])
        return cls(list(range(length)), **kwargs)
