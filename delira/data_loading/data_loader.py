import numpy as np
from delira.data_loading.dataset import AbstractDataset, DictDataset, \
    IterableDataset
from collections import Iterable, defaultdict


class DataLoader:
    """
    Basic Dataloader class, that returns data for a given set of indices and
    combines it as batches
    """

    def __init__(self, data):
        """
        Parameters
        ----------
        data : Any
            the data to use; Ideally this either is a dataset, an iterable or
            a dict, but in general, this must only be indexable, have a length
            and return a dict of arrays if indexed
        """
        self._process_id = None
        if isinstance(data, AbstractDataset):
            dataset = data

        else:
            # wrap it into dataset depending on datatype
            if isinstance(data, dict):
                dataset = DictDataset(data)
            elif isinstance(data, Iterable):
                dataset = IterableDataset(data)
            else:
                raise TypeError("Invalid dataset type: %s"
                                % type(data).__name__)

        self.dataset = dataset

    def __call__(self, indices):
        """
        Loads data for given indices and combines them to batches
        Parameters
        ----------
        indices : list
            a list of integers specifying the data indices
        Returns
        -------
        dict
            a dict of numpy arrays (specifying the batches)
        """

        # get data for all indices
        data = [self.dataset[idx] for idx in indices]

        data_dict = defaultdict(list)

        # concatenate dict entities by keys
        for _result_dict in data:
            for key, val in _result_dict.items():
                data_dict[key].append(val)

        # convert list to numpy arrays
        for key, val_list in data_dict.items():
            data_dict[key] = np.asarray(val_list)

        return data_dict

    @property
    def process_id(self):
        """
        A Property to access the process id
        Returns
        -------
        int
            the process id
        """
        if self._process_id is None:
            return 0
        return self._process_id

    @process_id.setter
    def process_id(self, new_id):
        """
        Setter for the :attr:`process_id`; Makes sure, that the process id is
        only set once
        Parameters
        ----------
        new_id : int
        Raises
        ------
        AttributeError
            if the process id has already been set once
        """
        if self._process_id is not None:
            raise AttributeError("Attribute 'process_id' can be set only once")

        self._process_id = new_id
