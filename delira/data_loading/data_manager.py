import logging
import numpy as np
import typing
from batchgenerators.dataloading import SlimDataLoaderBase, \
    MultiThreadedAugmenter
from torch.utils.data import ConcatDataset
from .dataset import AbstractDataset, BaseCacheDataset, BaseLazyDataset
from .data_loader import BaseDataLoader
from .load_utils import default_load_fn_2d
from .sampler import SequentialSampler
from ..utils.decorators import make_deprecated

logger = logging.getLogger(__name__)


class BaseDataManager(object):
    """
    Class to Handle Data
    Creates Dataset , Dataloader and BatchGenerator

    """

    def __init__(self, data, batch_size, n_process_augmentation,
                 transforms, sampler_cls=SequentialSampler,
                 data_loader_cls=None, dataset_cls=None,
                 load_fn=default_load_fn_2d, from_disc=True, **kwargs):
        """

        Parameters
        ----------
        data : str or Dataset
            if str: Path to data samples
            if dataset: Dataset
        batch_size : int
            Number of samples per batch
        n_process_augmentation : int
            Number of processes for augmentations
        transforms :
            Data transformations for augmentation
        sampler_cls : AbstractSampler
            class defining the sampling strategy
        data_loader_cls : subclass of SlimDataLoaderBase
            DataLoader class
        dataset_cls : subclass of AbstractDataset
            Dataset class
        load_fn : function
            function to load simple sample
        from_disc : bool
            whether or not to load data from disc just the time it is needed
        **kwargs :
            other keyword arguments (needed for dataloading and passed to
            dataset_cls)

        Raises
        ------
        AssertionError
            * `data_loader_cls` is not :obj:`None` and not a subclass of
            `SlimDataLoaderBase`
            * `dataset_cls` is not :obj:`None` and not a subclass of
            :class:`.AbstractDataset`

        See Also
        --------
        :class:`AbstractDataset`

        """
        self.batch_size = batch_size

        self.n_process_augmentation = n_process_augmentation
        self.transforms = transforms

        if data_loader_cls is None:
            logger.info("No DataLoader Class specified. Using BaseDataLoader")
            data_loader_cls = BaseDataLoader
        else:
            assert issubclass(data_loader_cls, SlimDataLoaderBase), \
                "dater_loader_cls must be subclass of SlimDataLoaderBase"

        self.data_loader_cls = data_loader_cls

        if isinstance(data, AbstractDataset):
            self.dataset = data
        else:

            if dataset_cls is None:
                if from_disc:
                    dataset_cls = BaseLazyDataset
                else:
                    dataset_cls = BaseCacheDataset

                logger.info("No DataSet Class specified. Using %s instead" %
                            dataset_cls.__name__)

            else:
                assert issubclass(dataset_cls, AbstractDataset), \
                    "dataset_cls must be subclass of AbstractDataset"

            self.dataset = dataset_cls(data, load_fn, **kwargs)

        self.sampler = sampler_cls.from_dataset(self.dataset)

    def get_batchgen(self, seed=1):
        """
        Create DataLoader and Batchgenerator

        Parameters
        ----------
        seed : int
            seed for Random Number Generator

        Returns
        -------
        MultiThreadedAugmenter
            Batchgenerator

        Raises
        ------
        AssertionError
            :attr:`BaseDataManager.n_batches` is smaller than or equal to zero

        """
        assert self.n_batches > 0

        data_loader = self.data_loader_cls(self.dataset,
                                           batch_size=self.batch_size,
                                           num_batches=self.n_batches,
                                           seed=seed,
                                           sampler=self.sampler
                                           )

        return MultiThreadedAugmenter(data_loader, self.transforms,
                                      self.n_process_augmentation,
                                      num_cached_per_queue=2,
                                      seeds=self.n_process_augmentation*[seed])

    def get_subset(self, indices):
        """
        Returns a Subset of the current datamanager based on given indices
        
        Parameters
        ----------
        indices : iterable
            valid indices to extract subset from current dataset

        Returns
        -------
        :class:`BaseDataManager`
            manager containing the subset
        
        """

        subset_kwargs = {
            "batch_size": self.batch_size,
            "n_process_augmentation": self.n_process_augmentation,
            "transforms": self.transforms,
            "sampler_cls": self.sampler.__class__,
            "data_loader_cls": self.data_loader_cls,
            "dataset_cls": None,
            "load_fn": None,
            "from_disc": True
        }

        return self.__class__(self.dataset.get_subset(indices), **subset_kwargs)

    @make_deprecated("BaseDataManager.get_subset")
    def train_test_split(self, *args, **kwargs):
        """
        Calls :method:`AbstractDataset.train_test_split` and returns 
        a manager for each subset with same configuration as current manager

        .. deprecation:: 0.3
            method will be removed in next major release

        Parameters
        ----------
        *args : 
            positional arguments for 
            ``sklearn.model_selection.train_test_split``
        **kwargs :
            keyword arguments for 
            ``sklearn.model_selection.train_test_split``
        
        """

        trainset, valset = self.dataset.train_test_split(*args, **kwargs)

        subset_kwargs = {
            "batch_size": self.batch_size,
            "n_process_augmentation": self.n_process_augmentation,
            "transforms": self.transforms,
            "sampler_cls": self.sampler.__class__,
            "data_loader_cls": self.data_loader_cls,
            "dataset_cls": None,
            "load_fn": None,
            "from_disc": True
        }

        train_mgr = self.__class__(trainset, **subset_kwargs)
        val_mgr = self.__class__(valset, **subset_kwargs)

        return train_mgr, val_mgr

    @property
    def n_samples(self):
        """
        Number of Samples

        Returns
        -------
        int
            Number of Samples

        """
        return len(self.sampler)

    @property
    def n_batches(self):
        """
        Returns Number of Batches based on batchsize,
        number of samples and number of processes

        Returns
        -------
        int
            Number of Batches

        Raises
        ------
        AssertionError
            :attr:`BaseDataManager.n_samples` is smaller than or equal to zero

        """
        assert self.n_samples > 0
        if self.n_process_augmentation == 1:
            n_batches = int(np.floor(self.n_samples / self.batch_size))
        elif self.n_process_augmentation > 1:
            if (self.n_samples / self.batch_size) < self.n_process_augmentation:
                self.n_process_augmentation = 1
                logger.warning('Too few samples for n_process_augmentation={}. '
                               'Forcing n_process_augmentation={} '
                               'instead'.format(self.n_process_augmentation, 1))
            n_batches = int(np.floor(self.n_samples / self.batch_size /
                                     self.n_process_augmentation))
        else:
            raise ValueError('Invalid value for n_process_augmentation')
        return n_batches


class ConcatDataManager(object):
    """
    Class to concatenate DataManagers

    """

    def __init__(self, datamanager=typing.List[BaseDataManager]):
        """

        Parameters
        ----------
        datamanager : list
            the datamanagers which should be concatenated
            (All attributes except the dataset are extracted 
            from the first manager inside the list)

        """

        self.dataset = ConcatDataset(
            [tmp.dataset for tmp in datamanager])

        self.data_loader_cls = datamanager[0].data_loader_cls

        self.batch_size = datamanager[0].batch_size
        self.n_process_augmentation = datamanager[0].n_process_augmentation
        self.transforms = datamanager[0].transforms
        self.sampler = datamanager[0].sampler.__class__.from_dataset(
            self.dataset
        )

    def get_batchgen(self, seed=1):
        """
        Create DataLoader and Batchgenerator

        Parameters
        ----------
        seed : int
            seed for Random Number Generator

        Returns
        -------
        MultiThreadedAugmenter
            Batchgenerator

        Raises
        ------
        AssertionError
            :attr:`ConcatDataManager.n_batches` is smaller than or equal to zero

        """
        assert self.n_batches > 0

        data_loader = self.data_loader_cls(self.dataset,
                                           batch_size=self.batch_size,
                                           num_batches=self.n_batches,
                                           seed=seed,
                                           sampler=self.sampler
                                           )

        return MultiThreadedAugmenter(data_loader, self.transforms,
                                      self.n_process_augmentation,
                                      num_cached_per_queue=2,
                                      seeds=self.n_process_augmentation*[seed])

    @property
    def n_samples(self):
        """
        Number of Samples

        Returns
        -------
        int
            Number of Samples

        """
        return len(self.sampler)

    @property
    def n_batches(self):
        """
        Returns Number of Batches based on batchsize,
        number of samples and number of processes

        Returns
        -------
        int
            Number of Batches

        Raises
        ------
        AssertionError
            :attr:`ConcatDataManager.n_samples` is smaller than or equal to zero

        """
        assert self.n_samples > 0

        if self.n_process_augmentation == 1:
            n_batches = int(np.floor(self.n_samples / self.batch_size))
        elif self.n_process_augmentation > 1:
            n_batches = int(np.floor(
                self.n_samples / self.batch_size / self.n_process_augmentation))
        else:
            raise ValueError('Invalid value for n_process')
        return n_batches
