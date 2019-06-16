import inspect
import logging

from batchgenerators.dataloading import MultiThreadedAugmenter, \
    SingleThreadedAugmenter, SlimDataLoaderBase
from batchgenerators.transforms import AbstractTransform

from multiprocessing import Queue
from queue import Full

from delira import get_current_debug_mode
from .data_loader import BaseDataLoader
from .dataset import AbstractDataset, BaseCacheDataset, BaseLazyDataset
from .load_utils import default_load_fn_2d
from .sampler import SequentialSampler, AbstractSampler
from ..utils.decorators import make_deprecated

logger = logging.getLogger(__name__)


class Augmenter(object):
    """
    Class wrapping ``MultiThreadedAugmentor`` and ``SingleThreadedAugmenter``
    to provide a uniform API and to disable multiprocessing/multithreading
    inside the dataloading pipeline

    """

    def __init__(self, data_loader: BaseDataLoader, transforms,
                 n_process_augmentation, sampler, sampler_queues: list,
                 num_cached_per_queue=2, seeds=None, **kwargs):
        """

        Parameters
        ----------
        data_loader : :class:`BaseDataLoader`
            the dataloader providing the actual data
        transforms : Callable or None
            the transforms to use. Can be single callable or None
        n_process_augmentation : int
            the number of processes to use for augmentation (only necessary if
            not in debug mode)
        sampler : :class:`AbstractSampler`
            the sampler to use; must be used here instead of inside the
            dataloader to avoid duplications and oversampling due to
            multiprocessing
        sampler_queues : list of :class:`multiprocessing.Queue`
            queues to pass the sample indices to the actual dataloader
        num_cached_per_queue : int
            the number of samples to cache per queue (only necessary if not in
            debug mode)
        seeds : int or list
            the seeds for each process (only necessary if not in debug mode)
        **kwargs :
            additional keyword arguments
        """

        self._batchsize = data_loader.batch_size

        # don't use multiprocessing in debug mode
        if get_current_debug_mode():
            augmenter = SingleThreadedAugmenter(data_loader, transforms)

        else:
            assert isinstance(n_process_augmentation, int)
            # no seeds are given -> use default seed of 1
            if seeds is None:
                seeds = 1

            # only an int is gien as seed -> replicate it for each process
            if isinstance(seeds, int):
                seeds = [seeds] * n_process_augmentation

            # avoid same seeds for all processes
            if any([seeds[0] == _seed for _seed in seeds[1:]]):
                for idx in range(len(seeds)):
                    seeds[idx] = seeds[idx] + idx

            augmenter = MultiThreadedAugmenter(
                data_loader, transforms,
                num_processes=n_process_augmentation,
                num_cached_per_queue=num_cached_per_queue,
                seeds=seeds,
                **kwargs)

        self._augmenter = augmenter
        self._sampler = sampler
        self._sampler_queues = sampler_queues
        self._queue_id = 0

    def __iter__(self):
        """
        Function returning an iterator

        Returns
        -------
        Augmenter
            self
        """
        return self

    def _next_queue(self):
        idx = self._queue_id
        self._queue_id = (self._queue_id + 1) % len(self._sampler_queues)
        return self._sampler_queues[idx]

    def __next__(self):
        """
        Function to sample and load the next batch

        Returns
        -------
        dict
            the next batch
        """
        idxs = self._sampler(self._batchsize)
        queue = self._next_queue()

        # dont't wait forever. Release this after short timeout and try again
        # to avoid deadlock
        while True:
            try:
                queue.put(idxs, timeout=0.2)
                break
            except Full:
                continue

        return next(self._augmenter)

    def next(self):
        """
        Function to sample and load

        Returns
        -------
        dict
            the next batch
        """
        return next(self)

    @staticmethod
    def __identity_fn(*args, **kwargs):
        """
        Helper function accepting arbitrary args and kwargs and returning
        without doing anything

        Parameters
        ----------
        *args
            keyword arguments
        **kwargs
            positional arguments

        """
        return

    def _fn_checker(self, function_name):
        """
        Checks if the internal augmenter has a given attribute and returns it.
        Otherwise it returns ``__identity_fn``

        Parameters
        ----------
        function_name : str
            the function name to check for

        Returns
        -------
        Callable
            either the function corresponding to the given function name or
            ``__identity_fn``

        """
        # same as:
        # if hasattr(self._augmenter, function_name):
        #     return getattr(self._augmenter, functionname)
        # else:
        #     return self.__identity_fn
        # but one less getattr call, because hasattr also calls getattr and
        # handles AttributeError
        try:
            return getattr(self._augmenter, function_name)
        except AttributeError:
            return self.__identity_fn

    @property
    def _start(self):
        """
        Property to provide uniform API of ``_start``

        Returns
        -------
        Callable
            either the augmenter's ``_start`` method (if available) or
            ``__identity_fn`` (if not available)
        """
        return self._fn_checker("_start")

    def restart(self):
        """
        Property to provide uniform API of ``restart``

        Returns
        -------
        Callable
            either the augmenter's ``restart`` method (if available) or
            ``__identity_fn`` (if not available)
        """
        return self._fn_checker("restart")

    def _finish(self):
        """
        Property to provide uniform API of ``_finish``

        Returns
        -------
        Callable
            either the augmenter's ``_finish`` method (if available) or
            ``__identity_fn`` (if not available)
        """
        ret_val = self._fn_checker("_finish")()
        for queue in self._sampler_queues:
            queue.close()
            queue.join_thread()

        return ret_val

    @property
    def num_batches(self):
        """
        Property returning the number of batches

        Returns
        -------
        int
            number of batches
        """
        if isinstance(self._augmenter, MultiThreadedAugmenter):
            return self._augmenter.generator.num_batches

        return self._augmenter.data_loader.num_batches

    @property
    def num_processes(self):
        """
        Property returning the number of processes to use for loading and
        augmentation

        Returns
        -------
        int
            number of processes to use for loading and
            augmentation

        """
        if isinstance(self._augmenter, MultiThreadedAugmenter):
            return self._augmenter.num_processes

        return 1

    def __del__(self):
        """
        Function defining what to do, if object should be deleted

        """
        self._finish()
        del self._augmenter


class BaseDataManager(object):
    """
    Class to Handle Data
    Creates Dataset , Dataloader and BatchGenerator

    """

    def __init__(self, data, batch_size, n_process_augmentation,
                 transforms, sampler_cls=SequentialSampler,
                 sampler_kwargs=None,
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
        sampler_kwargs : dict
            keyword arguments for sampling
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

        # Instantiate Hidden variables for property access
        if sampler_kwargs is None:
            sampler_kwargs = {}
        self._batch_size = None
        self._n_process_augmentation = None
        self._transforms = None
        self._data_loader_cls = None
        self._dataset = None
        self._sampler = None

        # set actual values to properties
        self.batch_size = batch_size

        self.n_process_augmentation = n_process_augmentation
        self.transforms = transforms

        if data_loader_cls is None:
            logger.info("No DataLoader Class specified. Using BaseDataLoader")
            data_loader_cls = BaseDataLoader
        else:
            assert inspect.isclass(data_loader_cls), \
                "data_loader_cls must be class not instance of class"
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

        assert inspect.isclass(sampler_cls) and issubclass(sampler_cls,
                                                           AbstractSampler)
        self.sampler = sampler_cls.from_dataset(self.dataset, **sampler_kwargs)

    def get_batchgen(self, seed=1):
        """
        Create DataLoader and Batchgenerator

        Parameters
        ----------
        seed : int
            seed for Random Number Generator

        Returns
        -------
        Augmenter
            Batchgenerator

        Raises
        ------
        AssertionError
            :attr:`BaseDataManager.n_batches` is smaller than or equal to zero

        """
        assert self.n_batches > 0

        sampler_queues = []

        for idx in range(self.n_process_augmentation):
            sampler_queues.append(Queue())

        data_loader = self.data_loader_cls(
            self.dataset,
            batch_size=self.batch_size,
            num_batches=self.n_batches,
            seed=seed,
            sampler_queues=sampler_queues
        )

        return Augmenter(data_loader, self.transforms,
                         self.n_process_augmentation,
                         sampler=self.sampler,
                         sampler_queues=sampler_queues,
                         num_cached_per_queue=2,
                         seeds=self.n_process_augmentation * [seed])

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

        return self.__class__(
            self.dataset.get_subset(indices),
            **subset_kwargs)

    def update_state_from_dict(self, new_state: dict):
        """
        Updates internal state and therfore the behavior from dict.
        If a key is not specified, the old attribute value will be used

        Parameters
        ----------
        new_state : dict
            The dict to update the state from.
            Valid keys are:

                * ``batch_size``
                * ``n_process_augmentation``
                * ``data_loader_cls``
                * ``sampler``
                * ``sampling_kwargs``
                * ``transforms``

            If a key is not specified, the old value of the corresponding
            attribute will be used

        Raises
        ------
        KeyError
            Invalid keys are specified

        """

        # update batch_size if specified
        self.batch_size = new_state.pop("batch_size", self.batch_size)
        # update n_process_augmentation if specified
        self.n_process_augmentation = new_state.pop(
            "n_process_augmentation", self.n_process_augmentation)
        # update data_loader_cls if specified
        self.data_loader_cls = new_state.pop("data_loader_cls",
                                             self.data_loader_cls)
        # update
        new_sampler = new_state.pop("sampler", None)
        if new_sampler is not None:
            self.sampler = new_sampler.from_dataset(
                self.dataset,
                **new_state.pop("sampling_kwargs", {}))
        self.transforms = new_state.pop("transforms", self.transforms)

        if new_state:
            raise KeyError("Invalid Keys in new_state given: %s"
                           % (','.join(map(str, new_state.keys()))))

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
    def batch_size(self):
        """
        Property to access the batchsize

        Returns
        -------
        int
            the batchsize
        """

        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_batch_size):
        """
        Setter for current batchsize, casts to int before setting the attribute

        Parameters
        ----------
        new_batch_size : int, Any
            the new batchsize; should be int but can be of any type that can be
            casted to an int

        """

        self._batch_size = int(new_batch_size)

    @property
    def n_process_augmentation(self):
        """
        Property to access the number of augmentation processes

        Returns
        -------
        int
            number of augmentation processes
        """

        if get_current_debug_mode():
            return 1
        return self._n_process_augmentation

    @n_process_augmentation.setter
    def n_process_augmentation(self, new_process_number):
        """
        Setter for number of augmentation processes, casts to int before
        setting the attribute


        Parameters
        ----------
        new_process_number : int, Any
            new number of augmentation processes; should be int but can be of
            any type that can be casted to an int

        """

        self._n_process_augmentation = int(new_process_number)

    @property
    def transforms(self):
        """
        Property to access the current data transforms

        Returns
        -------
        None, ``AbstractTransform``
            The transformation, can either be None or an instance of
            ``AbstractTransform``
        """

        return self._transforms

    @transforms.setter
    def transforms(self, new_transforms):
        """
        Setter for data transforms, assert if transforms are of valid type
        (either None or instance of ``AbstractTransform``)

        Parameters
        ----------
        new_transforms : None, ``AbstractTransform``
            the new transforms

        """

        assert new_transforms is None or isinstance(new_transforms,
                                                    AbstractTransform)

        self._transforms = new_transforms

    @property
    def data_loader_cls(self):
        """
        Property to access the current data loader class

        Returns
        -------
        type
            Subclass of ``SlimDataLoaderBase``
        """

        return self._data_loader_cls

    @data_loader_cls.setter
    def data_loader_cls(self, new_loader_cls):
        """
        Setter for current data loader class, asserts if class is of valid type
        (must be a class and a subclass of ``SlimDataLoaderBase``)

        Parameters
        ----------
        new_loader_cls : type
            the new data loader class

        """

        assert inspect.isclass(new_loader_cls) and issubclass(
            new_loader_cls, SlimDataLoaderBase)

        self._data_loader_cls = new_loader_cls

    @property
    def dataset(self):
        """
        Property to access the current dataset

        Returns
        -------
        :class:`AbstractDataset`
            the current dataset

        """

        return self._dataset

    @dataset.setter
    def dataset(self, new_dataset):
        """
        Setter for new dataset

        Parameters
        ----------
        new_dataset : :class:`AbstractDataset`

        """

        assert isinstance(new_dataset, AbstractDataset)
        self._dataset = new_dataset

    @property
    def sampler(self):
        """
        Property to access the current sampler

        Returns
        -------

        :class:`AbstractSampler`
            the current sampler
        """

        return self._sampler

    @sampler.setter
    def sampler(self, new_sampler):
        """
        Setter for current sampler.
        If a valid class instance is passed, the sampler is simply assigned, if
        a valid class type is passed, the sampler is created from the dataset

        Parameters
        ----------
        new_sampler : :class:`AbstractSampler`, type
            instance or class object of new sampler

        Raises
        ------
        ValueError
            Neither a valid class instance nor a valid class type is given

        """

        if inspect.isclass(new_sampler) and issubclass(new_sampler,
                                                       AbstractSampler):
            self._sampler = new_sampler.from_dataset(self.dataset)

        elif isinstance(new_sampler, AbstractSampler):
            self._sampler = new_sampler

        else:
            raise ValueError("Given Sampler is neither a subclass of \
                            AbstractSampler, nor an instance of a sampler ")

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
        Returns Number of Batches based on batchsize and number of samples

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

        n_batches = self.n_samples // self.batch_size

        truncated_batch = self.n_samples % self.batch_size

        n_batches += int(bool(truncated_batch))

        return n_batches
