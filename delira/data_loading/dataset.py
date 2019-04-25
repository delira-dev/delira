import abc
import os
import typing
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from delira import get_backends

from ..utils import subdirs
from ..utils.decorators import make_deprecated


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

    @make_deprecated("Dataset.get_subset")
    def train_test_split(self, *args, **kwargs):
        """
        split dataset into train and test data

        .. deprecated:: 0.3
            method will be removed in next major release

        Parameters
        ----------
        *args :
            positional arguments of ``train_test_split``
        **kwargs :
            keyword arguments of ``train_test_split``

        Returns
        -------
        :class:`BlankDataset`
            train dataset
        :class:`BlankDataset`
            test dataset

        See Also
        --------
        ``sklearn.model_selection.train_test_split``

        """

        train_idxs, test_idxs = train_test_split(
            np.arange(len(self)), *args, **kwargs)

        return self.get_subset(train_idxs), self.get_subset(test_idxs)


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
            assert os.path.isdir(path), '%s is not a valid directory' % dir
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
            assert os.path.isdir(path), '%s is not a valid directory' % dir
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
            if data_path is a list, _sample_fn is called for elements in the list
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


@make_deprecated('Will be removed in favour of LoadSample function.')
class Nii3DLazyDataset(BaseLazyDataset):
    """
       Dataset to load 3D medical images (e.g. from .nii files) during training
        """

    def __init__(self, data_path, load_fn, img_extensions, gt_extensions,
                 img_files, label_file, **load_kwargs):
        """
         Parameters
        ----------
        data_path : str
            root path to data samples where each samples has it's own folder
        load_fn : function
            function to load single data sample
        img_extensions : list
            valid extensions of image files
        gt_extensions : list
            valid extensions of label files
        img_files : list
            list of image filenames
        label_file : string
            label file name
        **load_kwargs :
            additional loading keyword arguments (image shape,
            channel number, ...); passed to load_fn
         """
        self.img_files = img_files
        self.label_file = label_file
        super().__init__(data_path, load_fn, **load_kwargs)

    def _make_dataset(self, path):
        """
        Helper Function to make a dataset containing all samples in a certain
        directory
         Parameters
        ----------
        path: str
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
        assert os.path.isdir(path)

        data = [[{'img': [os.path.join(t, i) for i in self.img_files],
                  'label': os.path.join(t, self.label_file)}]
                for t in subdirs(path)]
        return data


@make_deprecated('Will be removed in favour of LoadSample function.')
class Nii3DCacheDatset(BaseCacheDataset):
    """
    Dataset to load 3D medical images (e.g. from .nii files) before training
     """

    def __init__(self, data_path, load_fn, img_extensions, gt_extensions,
                 img_files, label_file, **load_kwargs):
        """
         Parameters
        ----------
        data_path : str
            root path to data samples where each samples has it's own folder
        load_fn : function
            function to load single data sample
        img_extensions : list
            valid extensions of image files
        gt_extensions : list
            valid extensions of label files
        img_files : list
            list of image filenames
        label_file : str
            label file name
        **load_kwargs :
            additional loading keyword arguments (image shape,
            channel number, ...); passed to load_fn
         """
        self.img_files = img_files
        self.label_file = label_file
        super().__init__(data_path, load_fn, **load_kwargs)

    def _make_dataset(self, path):
        """
        Helper Function to make a dataset containing all samples in a certain
        directory
         Parameters
        ----------
        path: str
            path to data samples
         Returns
        -------
        list
            list of samples
         Raises
        ------
        AssertionError
            if `path` is not a valid directory
         """
        assert os.path.isdir(path)
        data = []
        for s in tqdm(subdirs(path), unit='samples', desc="Loading samples"):
            files = {'img': [os.path.join(s, i) for i in self.img_files],
                     'label': os.path.join(s, self.label_file)}

            data.append(self._load_fn(files, **self._load_kwargs))
        return data


if "TORCH" in get_backends():
    from torchvision.datasets import CIFAR10, CIFAR100, EMNIST, MNIST, FashionMNIST


    class TorchvisionClassificationDataset(AbstractDataset):
        """
        Wrapper for torchvision classification datasets to provide consistent API

        """

        def __init__(self, dataset, root="/tmp/", train=True, download=True,
                     img_shape=(28, 28), one_hot=False, **kwargs):
            """

            Parameters
            ----------
            dataset : str
                Defines the dataset to use.
                must be one of
                ['mnist', 'emnist', 'fashion_mnist', 'cifar10', 'cifar100']
            root : str
                path dataset (If download is True: dataset will be extracted here;
                else: path to extracted dataset)
            train : bool
                whether to use the train or the testset
            download : bool
                whether or not to download the dataset
                (If already downloaded at specified path,
                it won't be downloaded again)
            img_shape : tuple
                Height and width of output images (will be interpolated)
            **kwargs :
                Additional keyword arguments passed to the torchvision dataset
                class for initialization

            """
            super().__init__("", None)

            self.download = download
            self.train = train
            self.root = root
            self.img_shape = img_shape
            self.num_classes = None
            self.one_hot = one_hot
            self.data = self._make_dataset(dataset, **kwargs)

        def _make_dataset(self, dataset, **kwargs):
            """
            Create the actual dataset

            Parameters
            ----------
            dataset: str
                Defines the dataset to use.
                must be one of
                ['mnist', 'emnist', 'fashion_mnist', 'cifar10', 'cifar100']
            **kwargs :
                Additional keyword arguments passed to the torchvision dataset
                class for initialization

            Returns
            -------
            torchvision.Dataset
                actual Dataset

            Raises
            ------
            KeyError
                Dataset string does not specify a valid dataset

            """
            if dataset.lower() == "mnist":
                _dataset_cls = MNIST
                self.num_classes = 10
            elif dataset.lower() == "emnist":
                _dataset_cls = EMNIST
                # TODO: EMNIST requires split as kwarg. Search for 'split' in kwargs and
                # update self.num_classes accordingly
                # https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.EMNIST
                self.num_classes = None
            elif dataset.lower() == "fashion_mnist":
                _dataset_cls = FashionMNIST
                self.num_classes = 10
            elif dataset.lower() == "cifar10":
                _dataset_cls = CIFAR10
                self.num_classes = 10
            elif dataset.lower() == "cifar100":
                _dataset_cls = CIFAR100
                self.num_classes = 100
            else:
                raise KeyError("Dataset %s not found!" % dataset.lower())

            return _dataset_cls(root=self.root, train=self.train,
                                download=self.download, **kwargs)

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

            data = self.data[index]
            data_dict = {"data": np.array(data[0]),
                         "label": data[1].reshape(1).astype(np.float32)}

            if self.one_hot:
                # TODO: Remove and refer to batchgenerators transform:
                # https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/transforms/utility_transforms.py#L97
                def make_onehot(num_classes, labels):
                    """
                    Function that converts label-encoding to one-hot format.

                    Parameters
                    ----------
                    num_classes : int
                        number of classes present in the dataset

                    labels : np.ndarray
                        labels in label-encoding format

                    Returns
                    -------
                    np.ndarray
                        labels in one-hot format
                    """
                    if isinstance(labels, list) or isinstance(labels, int):
                        labels = np.asarray(labels)
                    assert isinstance(labels, np.ndarray)
                    if len(labels.shape) > 1:
                        one_hot = np.zeros(shape=(list(labels.shape) + [num_classes]),
                                           dtype=labels.dtype)
                        for i, c in enumerate(np.arange(num_classes)):
                            one_hot[..., i][labels == c] = 1
                    else:
                        one_hot = np.zeros(shape=([num_classes]),
                                           dtype=labels.dtype)
                        for i, c in enumerate(np.arange(num_classes)):
                            if labels == c:
                                one_hot[i] = 1
                    return one_hot

                data_dict['label'] = make_onehot(self.num_classes, data_dict['label'])

            img = data_dict["data"]

            img = resize(img, self.img_shape,
                         mode='reflect', anti_aliasing=True)
            if len(img.shape) <= 3:
                img = img.reshape(
                    *img.shape, 1)

            img = img.transpose(
                (len(img.shape) - 1, *range(len(img.shape) - 1)))

            data_dict["data"] = img.astype(np.float32)
            return data_dict

        def __len__(self):
            """
            Return Number of samples

            Returns
            -------
            int
                number of samples

            """
            return len(self.data)
