import abc
import os
import typing

import numpy as np
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from collections import Iterable
from tqdm import tqdm

from delira.utils import subdirs


class AbstractDataset:
    """
    Base Class for Dataset

    """

    def __init__(self, data_path: str, load_fn: typing.Callable):
        """

        Parameters
        ----------
        data_path : str
            path to data samples
        load_fn : function
            function to load single sample
        """
        self.data_path = data_path
        self._load_fn = load_fn
        self.data = []

    @abc.abstractmethod
    def _make_dataset(self, path: str):
        """
        Create dataset

        Parameters
        ----------
        path : str
            path to data samples

        Returns
        -------
        list
            data: List of sample paths if lazy; List of samples if not

        """
        pass

    @abc.abstractmethod
    def __getitem__(self, index):
        """
        return data with given index (and loads it before if lazy)

        Parameters
        ----------
        index : int
            index of data

        Returns
        -------
        dict
            data

        """
        pass

    def __len__(self):
        """
        Return number of samples

        Returns
        -------
        int
            number of samples
        """
        return len(self.data)

    def __iter__(self):
        """
        Return an iterator for the dataset

        Returns
        -------
        object
            a single sample
        """
        return _DatasetIter(self)

    def get_sample_from_index(self, index):
        """
        Returns the data sample for a given index
        (without any loading if it would be necessary)
        This implements the base case and can be subclassed
        for index mappings.
        The actual loading behaviour (lazy or cached) should be
        implemented in ``__getitem__``

        See Also
        --------
        :method:ConcatDataset.get_sample_from_index
        :method:BaseLazyDataset.__getitem__
        :method:BaseCacheDataset.__getitem__

        Parameters
        ----------
        index : int
            index corresponding to targeted sample

        Returns
        -------
        Any
            sample corresponding to given index
        """

        return self.data[index]

    def get_subset(self, indices):
        """
        Returns a Subset of the current dataset based on given indices

        Parameters
        ----------
        indices : iterable
            valid indices to extract subset from current dataset

        Returns
        -------
        :class:`BlankDataset`
            the subset

        """

        # extract other important attributes from current dataset
        kwargs = {}

        for key, val in vars(self).items():
            if not (key.startswith("__") and key.endswith("__")):

                if key == "data":
                    continue
                kwargs[key] = val

        kwargs["old_getitem"] = self.__class__.__getitem__
        subset_data = [self.get_sample_from_index(idx) for idx in indices]

        return BlankDataset(subset_data, **kwargs)


class _DatasetIter(object):
    """
    Iterator for dataset
    """

    def __init__(self, dset):
        """

        Parameters
        ----------
        dset: :class: `AbstractDataset`
            the dataset which should be iterated
        """
        self._dset = dset
        self._curr_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._curr_index >= len(self._dset):
            raise StopIteration

        sample = self._dset[self._curr_index]
        self._curr_index += 1
        return sample


class DictDataset(AbstractDataset):
    """
    Dataset to wrap a dict of keys and iterables.
    """

    def __init__(self, data: dict):
        """

        Parameters
        ----------
        data : dict
            dictionary consisting of keys and iterables.
            The iterables should contain an item for each index
        """
        super().__init__(None, None)
        self._data = data

    def __getitem__(self, index: int):
        """
        Function to make the dataset indexable. Returns the sample
        corresponding to the given index

        Parameters
        ----------
        index : int
            the index specifying the sample to return

        Returns
        -------
        dict
            the sample corresponding to :param:`index`

        """
        return {k: v[index] for k, v in self._data.items()}

    def get_sample_from_index(self, index):
        """
        Mapping from index to sample

        Parameters
        ----------
        index : int
            the index specifying the sample to return

        Returns
        -------
        dict
            the sample corresponding to :param:`index`

        """
        return self[index]

    def _make_dataset(self, path: str):
        """
        Function to create the dataset
        (not necessary here, since the data is already in memory)

        Parameters
        ----------
        path : str
            the path to load the data from

        """
        pass

    def __len__(self):
        """
        Function to determine the dataset's length

        Returns
        -------
        int
            the number of samples
        """
        return min([len(v) for v in self._data.values()])


class IterableDataset(AbstractDataset):
    """
    Dataset to wrap a list of dicts.
    """

    def __init__(self, data: Iterable):
        """

        Parameters
        ----------
        data : Iterable
            an iterable of dicts each representing a single sample
        """
        super().__init__(None, None)
        self._data = data

    def __getitem__(self, index):
        """
        Function to make the dataset indexable. Returns the sample
        corresponding to the given index

        Parameters
        ----------
        index : int
           the index specifying the sample to return

        Returns
        -------
        dict
           the sample corresponding to :param:`index`

       """
        return self._data[index]

    def get_sample_from_index(self, index):
        """
        Mapping from index to sample

        Parameters
        ----------
        index : int
            the index specifying the sample to return

        Returns
        -------
        dict
            the sample corresponding to :param:`index`

        """
        return self[index]

    def _make_dataset(self, path: str):
        """
        Function to create the dataset
        (not necessary here, since the data is already in memory)

        Parameters
        ----------
        path : str
            the path to load the data from

        """
        pass

    def __len__(self):
        """
        Function to determine the dataset's length

        Returns
        -------
        int
            the number of samples
        """
        return len(self._data)


class BlankDataset(AbstractDataset):
    """
    Blank Dataset loading the data, which has been passed
    in it's ``__init__`` by it's ``_sample_fn``

    """

    def __init__(self, data, old_getitem, **kwargs):
        """

        Parameters
        ----------
        data : iterable
            data to load
        old_getitem : function
            get item method of previous dataset
        **kwargs :
            additional keyword arguments (are set as class attribute)

        """
        super().__init__(None, None)

        self.data = data
        self._old_getitem = old_getitem

        for key, val in kwargs.items():
            setattr(self, key, val)

    def __getitem__(self, index):
        """
        returns single sample corresponding to ``index`` via the ``_sample_fn``

        Parameters
        ----------
        index : int
            index specifying the data to load

        Returns
        -------
        dict
            dictionary containing a single sample

        """
        return self._old_getitem(self, index)

    def __len__(self):
        """
        returns the length of the dataset

        Returns
        -------
        int
            number of samples

        """
        return len(self.data)


class BaseCacheDataset(AbstractDataset):
    """
    Dataset to preload and cache data

    Notes
    -----
    data needs to fit completely into RAM!

    """

    def __init__(self, data_path: typing.Union[str, list],
                 load_fn: typing.Callable, **load_kwargs):
        """

        Parameters
        ----------
        data_path : str or list
            if data_path is a string, _sample_fn is called for all items inside
            the specified directory
            if data_path is a list, _sample_fn is called for elements in the
            list
        load_fn : function
            function to load a single data sample
        **load_kwargs :
            additional loading keyword arguments (image shape,
            channel number, ...); passed to _sample_fn

        """
        super().__init__(data_path, load_fn)
        self._load_kwargs = load_kwargs
        self.data = self._make_dataset(data_path)

    def _make_dataset(self, path: typing.Union[str, list]):
        """
        Helper Function to make a dataset containing all samples in a certain
        directory

        Parameters
        ----------
        path: str or list
            if data_path is a string, _sample_fn is called for all items inside
            the specified directory
            if data_path is a list, _sample_fn is called for elements in the
            list

        Returns
        -------
        list
            list of items which where returned from _sample_fn (typically dict)

        Raises
        ------
        AssertionError
            if `path` is not a list and is not a valid directory

        """
        data = []
        if isinstance(path, list):
            # iterate over all elements
            for p in tqdm(path, unit='samples', desc="Loading samples"):
                data.append(self._load_fn(p, **self._load_kwargs))
        else:
            # call _sample_fn for all elements inside directory
            assert os.path.isdir(path), '%s is not a valid directory' % path
            for p in tqdm(os.listdir(path), unit='samples',
                          desc="Loading samples"):
                data.append(self._load_fn(os.path.join(path, p),
                                          **self._load_kwargs))
        return data

    def __getitem__(self, index):
        """
        return data sample specified by index

        Parameters
        ----------
        index : int
            index to specifiy which data sample to return

        Returns
        -------
        dict
            data sample

        """
        data_dict = self.get_sample_from_index(index)
        return data_dict


class BaseLazyDataset(AbstractDataset):
    """
    Dataset to load data in a lazy way

    """

    def __init__(self, data_path: typing.Union[str, list],
                 load_fn: typing.Callable, **load_kwargs):
        """

        Parameters
        ----------
        data_path : str or list
            if data_path is a string, _sample_fn is called for all items inside
            the specified directory
            if data_path is a list, _sample_fn is called for elements in the
            list
        load_fn : function
            function to load single data sample
        **load_kwargs :
            additional loading keyword arguments (image shape,
            channel number, ...); passed to _sample_fn

        """
        super().__init__(data_path, load_fn)
        self._load_kwargs = load_kwargs
        self.data = self._make_dataset(self.data_path)

    def _make_dataset(self, path: typing.Union[str, list]):
        """
        Helper Function to make a dataset containing paths to all images in a
        certain directory

        Parameters
        ----------
        path : str or list
            path to data samples

        Returns
        -------
        list
            list of sample paths

        Raises
        ------
        AssertionError
            if `path` is not a valid directory

        """
        if isinstance(path, list):
            # generate list from iterable
            data = list(path)
        else:
            # generate list from all items
            assert os.path.isdir(path), '%s is not a valid directory' % path
            data = [os.path.join(path, p) for p in os.listdir(path)]
        return data

    def __getitem__(self, index):
        """
        load data sample specified by index

        Parameters
        ----------
        index : int
            index to specifiy which data sample to load

        Returns
        -------
        dict
            loaded data sample
        """
        data_dict = self._load_fn(self.get_sample_from_index(index),
                                  **self._load_kwargs)
        return data_dict


class BaseExtendCacheDataset(BaseCacheDataset):
    """
    Dataset to preload and cache data. Function to load sample is expected
    to return an iterable which can contain multiple samples

    Notes
    -----
    data needs to fit completely into RAM!

    """

    def __init__(self, data_path: typing.Union[str, list],
                 load_fn: typing.Callable, **load_kwargs):
        """

        Parameters
        ----------
        data_path : str or list
            if data_path is a string, _sample_fn is called for all items inside
            the specified directory
            if data_path is a list, _sample_fn is called for elements in the
            list
        load_fn : function
            function to load a multiple data samples at once. Needs to return
            an iterable which extends the internal list.
        **load_kwargs :
            additional loading keyword arguments (image shape,
            channel number, ...); passed to _sample_fn

        See Also
        --------
        :class: `BaseCacheDataset`

        """
        super().__init__(data_path, load_fn, **load_kwargs)

    def _make_dataset(self, path: typing.Union[str, list]):
        """
        Helper Function to make a dataset containing all samples in a certain
        directory

        Parameters
        ----------
        path: str or iterable
            if data_path is a string, _sample_fn is called for all items inside
            the specified directory
            if data_path is a list, _sample_fn is called for elements in the
            list

        Returns
        -------
        list
            list of items which where returned from _sample_fn (typically dict)

        Raises
        ------
        AssertionError
            if `path` is not a list and is not a valid directory

        """
        data = []
        if isinstance(path, list):
            # iterate over all elements
            for p in tqdm(path, unit='samples', desc="Loading samples"):
                data.extend(self._load_fn(p, **self._load_kwargs))
        else:
            # call _sample_fn for all elements inside directory
            assert os.path.isdir(path), '%s is not a valid directory' % dir
            for p in tqdm(os.listdir(path), unit='samples',
                          desc="Loading samples"):
                data.extend(self._load_fn(os.path.join(path, p),
                                          **self._load_kwargs))
        return data


class ConcatDataset(AbstractDataset):
    def __init__(self, *datasets):
        """
        Concatenate multiple datasets to one

        Parameters
        ----------
        datasets:
            variable number of datasets
        """
        super().__init__(None, None)

        # TODO: Why should datasets[0] be a list not a AbstractDataset?

        # check if first item in datasets is list and datasets is of length 1
        if (len(datasets) == 1) and isinstance(datasets[0], list):
            datasets = datasets[0]

        self.data = datasets

    def get_sample_from_index(self, index):
        """
        Returns the data sample for a given index
        (without any loading if it would be necessary)
        This method implements the index mapping of a global index to
        the subindices for each dataset.
        The actual loading behaviour (lazy or cached) should be
        implemented in ``__getitem__``

        See Also
        --------
        :method:AbstractDataset.get_sample_from_index
        :method:BaseLazyDataset.__getitem__
        :method:BaseCacheDataset.__getitem__

        Parameters
        ----------
        index : int
            index corresponding to targeted sample

        Returns
        -------
        Any
            sample corresponding to given index
        """

        curr_max_index = 0
        for dset in self.data:
            prev_max_index = curr_max_index
            curr_max_index += len(dset)

            if prev_max_index <= index < curr_max_index:
                return dset[index - prev_max_index]

            else:
                continue

        raise IndexError("Index %d is out of range for %d items in datasets" %
                         (index, len(self)))

    def __getitem__(self, index):
        return self.get_sample_from_index(index)

    def __len__(self):
        return sum([len(dset) for dset in self.data])
