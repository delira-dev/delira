import multiprocessing
from multiprocessing import connection as mpconnection
from collections import Callable
import abc
import os
import sys
import numpy as np
import random

from delira.data_loading.sampler import AbstractSampler, BatchSampler
from delira.data_loading.data_loader import DataLoader
from delira import get_current_debug_mode


class AbstractAugmenter(object):
    """
    Basic Augmenter Class providing a general Augmenter API
    """

    def __init__(
            self,
            data_loader,
            batchsize,
            sampler,
            transforms=None,
            seed=1,
            drop_last=False):
        """
        Parameters
        ----------
        data_loader : :class:`DataLoader`
            the dataloader, loading samples for given indices
        batchsize : int
            the batchsize to use for sampling
        sampler : :class:`AbstractSampler`
            the sampler_old (may be batch sampler_old or usual sampler_old),
            defining the actual sampling strategy; Is an iterable yielding
            indices
        transforms : :class:`collections.Callable`
            the transforms to apply; defaults to None
        seed : int
            the basic seed; default: 1
        drop_last : bool
            whether to drop the last (possibly smaller) batch or not
        """

        self._data_loader = data_loader

        if not isinstance(sampler, BatchSampler):
            if isinstance(sampler, AbstractSampler):
                sampler = BatchSampler(sampler, batchsize,
                                       drop_last=drop_last)
            else:
                raise TypeError("Invalid Sampler given: %s" % str(sampler))

        self._sampler = sampler

        self._drop_last = drop_last

        self._transforms = transforms
        self._seed = seed

        # seed numpy.random and random as these are the random number
        # generators, which might be used for sampling
        np.random.seed(seed)
        random.seed(seed)

    @abc.abstractmethod
    def __iter__(self):
        raise NotImplementedError


class _ParallelAugmenter(AbstractAugmenter):
    """
    An Augmenter that loads and augments multiple batches in parallel
    """

    def __init__(self, data_loader, batchsize, sampler, num_processes=None,
                 transforms=None, seed=1, drop_last=False):
        """
        Parameters
        ----------
        data_loader : :class:`DataLoader`
            the dataloader, loading samples for given indices
         batchsize : int
            the batchsize to use for sampling
        sampler : :class:`AbstractSampler`
            the sampler_old (may be batch sampler_old or usual sampler_old),
            defining the actual sampling strategy; Is an iterable yielding
            indices
        num_processes : int
            the number of processes to use for dataloading + augmentation;
            if None: the number of available CPUs will be used as number of
            processes
        transforms : :class:`collections.Callable`
            the transforms to apply; defaults to None
        seed : int
            the basic seed; default: 1
        drop_last : bool
            whether to drop the last (possibly smaller) batch or not
        """

        super().__init__(data_loader, batchsize, sampler, transforms, seed,
                         drop_last)

        if num_processes is None:
            num_processes = os.cpu_count()

        self._num_processes = num_processes

        self._processes = []

        self._index_pipes = []
        self._data_pipes = []

        self._index_pipe_counter = 0
        self._data_pipe_counter = 0
        self._abort_event = None
        self._data_queued = []
        self._processes_running = False

    @property
    def abort_event(self):
        """
        Property to access the abortion Event

        Returns
        -------
        :class:`multiprocessing.Event`
            the abortion event
        """
        return self._abort_event

    @abort_event.setter
    def abort_event(self, new_event):
        """
        Setter for the abortion Event;

        Parameters
        ----------
        new_event : class:`multiprocessing.Event`
            the new event
        """

        self._abort_event = new_event

    def _start_processes(self):
        """
        Starts new processes and pipes for interprocess communication
        """

        # reset abortion event
        self.abort_event = multiprocessing.Event()

        # for each process do:
        for i in range(self._num_processes):
            # start two oneway pipes (one for passing index to workers
            # and one for passing back data to main process)
            recv_conn_out, send_conn_out = multiprocessing.Pipe(duplex=False)
            recv_conn_in, send_conn_in = multiprocessing.Pipe(duplex=False)

            # create the actual process
            process = _WorkerProcess(dataloader=self._data_loader,
                                     output_pipe=send_conn_out,
                                     index_pipe=recv_conn_in,
                                     transforms=self._transforms,
                                     abort_event=self._abort_event,
                                     process_id=i)
            process.daemon = True
            process.start()
            # wait until process was created and started
            while not process.is_alive():
                pass

            # append process and pipes to list
            self._processes.append(process)
            self._index_pipes.append(send_conn_in),
            self._data_pipes.append(recv_conn_out)
            self._data_queued.append(0)
            self._processes_running = True

    def _shutdown_processes(self):
        """
        Shuts down the processes and resets all related flags and counters
        """

        # create copy to avoid modifying the list we iterate over
        worker = list(
            zip(self._data_pipes, self._index_pipes, self._processes))

        for _data_conn, _index_conn, _process in worker:

            _index_conn.send(None)

            _process.join()
            if sys.version_info >= (3, 7):
                _process.close()
            else:
                _process.terminate()

            _index_conn.close()
            _data_conn.close()

            self._data_pipes.pop()
            self._data_queued.pop()
            self._index_pipes.pop()
            self._processes.pop()

        # reset running process flag and counters
        self._processes_running = False
        self._data_pipe_counter = 0
        self._index_pipe_counter = 0

    @property
    def _next_index_pipe(self):
        """
        Property implementing switch to next index pipe
        """
        ctr = self._index_pipe_counter
        new_ctr = (self._index_pipe_counter + 1) % self._num_processes
        self._index_pipe_counter = new_ctr

        return ctr

    @property
    def _next_data_pipe(self):
        """
        Property implementing switch to next data pipe
        """
        ctr = self._data_pipe_counter
        new_ctr = (self._data_pipe_counter + 1) % self._num_processes
        self._data_pipe_counter = new_ctr

        return ctr

    def _enqueue_indices(self, sample_idxs):
        """
        Enqueues a set of indices to workers while iterating over workers in
        cyclic way
        Parameters
        ----------
        sample_idxs : list
            the indices to enqueue to the workers
        """

        # iterating over all batch indices
        for idxs in sample_idxs:
            # switch to next counter
            index_pipe_ctr = self._next_index_pipe
            # increase number of queued batches for current worker
            self._data_queued[index_pipe_ctr] += 1
            # enqueue indices to worker
            self._index_pipes[index_pipe_ctr].send(idxs)

    def _receive_data(self):
        """
        Receives data from worker
        """
        # switching to next worker
        _data_pipe = self._next_data_pipe

        # receive data from worker
        data = self._data_pipes[_data_pipe].recv()
        # decrease number of enqueued batches for current worker
        self._data_queued[_data_pipe] -= 1

        return data

    def __iter__(self):
        self._start_processes()

        sampler_iter = iter(self._sampler)
        all_sampled = False

        try:
            # start by enqueuing two items per process as buffer
            _indices = []
            try:
                for i in range(self._num_processes * 2):
                    idxs = next(sampler_iter)
                    _indices.append(idxs)
            except StopIteration:
                all_sampled = True

            self._enqueue_indices(_indices)

            # iterate while not all data has been sampled and any data is
            # enqueued
            while True:

                if self.abort_event.is_set():
                    raise RuntimeError("Abort Event was set in one of the "
                                       "workers")

                # enqueue additional indices if sampler_old was not already
                # exhausted
                try:
                    if not all_sampled:
                        idxs = next(sampler_iter)
                        self._enqueue_indices([idxs])
                except StopIteration:
                    all_sampled = True

                # receive data from workers
                if any(self._data_queued):
                    yield self._receive_data()
                else:
                    break

        except Exception as e:
            # set abort event to shutdown workers
            self._abort_event.set()
            raise e

        finally:
            if self._processes_running:
                self._shutdown_processes()


class _WorkerProcess(multiprocessing.Process):
    """
    A Process running an infinite loop of loading data for given indices
    """

    def __init__(self, dataloader: DataLoader,
                 output_pipe: mpconnection.Connection,
                 index_pipe: mpconnection.Connection,
                 abort_event: multiprocessing.Event,
                 transforms: Callable,
                 process_id):
        """
        Parameters
        ----------
        dataloader : :class:`DataLoader`
            the data loader which loads the data corresponding to the given
            indices
        output_pipe : :class:`multiprocessing.connection.Connection`
            the pipe, the loaded data shoud be sent to
        index_pipe : :class:`multiprocessing.connection.Connection`
            the pipe to accept the indices
        abort_event : class:`multiprocessing.Event`
            the abortion event; will be set for every Exception;
            If set: Worker terminates
        transforms : :class:`collections.Callable`
            the transforms to transform the data
        process_id : int
            the process id
        """
        super().__init__()

        self._data_loader = dataloader
        self._output_pipe = output_pipe
        self._input_pipe = index_pipe
        self._abort_event = abort_event
        self._process_id = process_id
        self._transforms = transforms

    def run(self) -> None:
        # set the process id
        self._data_loader.process_id = self._process_id

        try:
            while True:
                # check if worker should terminate
                if self._abort_event.is_set():
                    raise RuntimeError("Abort Event has been set externally")

                # get indices if available (with timeout to frequently check
                # for abortions
                if self._input_pipe.poll(timeout=0.2):
                    idxs = self._input_pipe.recv()

                    # final indices -> shutdown workers
                    if idxs is None:
                        break

                    # load data
                    data = self._data_loader(idxs)

                    #
                    if self._transforms is not None:
                        data = self._transforms(**data)

                    self._output_pipe.send(data)

        except Exception as e:
            self._abort_event.set()
            raise e


class _SequentialAugmenter(AbstractAugmenter):
    """
    An Augmenter that loads and augments batches sequentially without any
    parallelism
    """

    def __init__(
            self,
            data_loader,
            batchsize,
            sampler,
            transforms=None,
            seed=1,
            drop_last=False):
        """
        Parameters
        ----------
        data_loader : :class:`DataLoader`
            the dataloader, loading samples for given indices
        sampler : :class:`AbstractSampler`
            the sampler_old (may be batch sampler_old or usual sampler_old),
            defining the actual sampling strategy; Is an iterable yielding
            indices
        transforms : :class:`collections.Callable`
            the transforms to apply; defaults to None
        seed : int
            the basic seed; default: 1
        drop_last : bool
            whether to drop the last (possibly smaller) batch or not
        """
        super().__init__(data_loader=data_loader, batchsize=batchsize,
                         sampler=sampler, transforms=transforms, seed=seed,
                         drop_last=drop_last)

    def __iter__(self):
        # create sampler_old iterator
        sampler_iter = iter(self._sampler)

        # for every index load and augment the data
        for idxs in sampler_iter:

            # load data
            data = self._data_loader(idxs)

            # transform data if transforms given
            if self._transforms is not None:
                data = self._transforms(**data)

            yield data


class Augmenter(object):
    """
    The actual Augmenter wrapping the :class:`_SequentialAugmenter` and the
    :class:`_ParallelAugmenter` and switches between them by arguments and
    debug mode
    """

    def __init__(self, data_loader, batchsize, sampler, num_processes=None,
                 transforms=None, seed=1, drop_last=False):
        """
        Parameters
        ----------
        data_loader : :class:`DataLoader`
            the dataloader, loading samples for given indices
        sampler : :class:`AbstractSampler`
            the sampler_old (may be batch sampler_old or usual sampler_old),
            defining the actual sampling strategy; Is an iterable yielding
            indices
        num_processes : int
            the number of processes to use for dataloading + augmentation;
            if None: the number of available CPUs will be used as number of
            processes
        transforms : :class:`collections.Callable`
            the transforms to apply; defaults to None
        seed : int
            the basic seed; default: 1
        drop_last : bool
            whether to drop the last (possibly smaller) batch or not
        """

        self._augmenter = self._resolve_augmenter_cls(num_processes,
                                                      data_loader=data_loader,
                                                      batchsize=batchsize,
                                                      sampler=sampler,
                                                      transforms=transforms,
                                                      seed=seed,
                                                      drop_last=drop_last)

    @staticmethod
    def _resolve_augmenter_cls(num_processes, **kwargs):
        """
        Resolves the augmenter class by the number of specified processes and
        the debug mode and creates an instance of the chosen class
        Parameters
        ----------
        num_processes : int
            the number of processes to use for dataloading + augmentation;
            if None: the number of available CPUs will be used as number of
            processes
        **kwargs :
            additional keyword arguments, used for instantiation of the chosen
            class
        Returns
        -------
        :class:`AbstractAugmenter`
            an instance of the chosen augmenter class
        """
        if get_current_debug_mode() or num_processes == 0:
            return _SequentialAugmenter(**kwargs)
        return _ParallelAugmenter(num_processes=num_processes, **kwargs)

    def __iter__(self):
        """
        Makes the Augmenter iterable by generators
        Returns
        -------
        Generator
            a generator function yielding the arguments
        """
        yield from self._augmenter
