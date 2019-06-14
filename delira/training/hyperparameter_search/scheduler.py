from delira.training.hyperparameter_search.reporter import ReporterCallback
from abc import abstractmethod
from delira.training.hyperparameter_search.utils import DebugSafePool
from delira.training.hyperparameter_search.search_algorithm import \
    SearchAlgorithm
from functools import partial
from asyncio import Lock, Semaphore
import numpy as np


def _concat_dict_items(dict_like):

    """
    Concatenates all items of same key across all dictionaries

    Parameters
    ----------
    dict_like

    Returns
    -------

    """
    for k, v in dict_like.items():
        if isinstance(v, dict):
            v = _concat_dict_items(v)
        else:
            v = np.concatenate(v)

        dict_like[k] = v

        return dict_like


class Scheduler:
    """
    An abstract Scheduler, which schedules different trials across different
    GPUs (currently only on same machine)
    """
    def __init__(self, run_fn, search_algo: SearchAlgorithm, gpus: list = None,
                 **kwargs):
        """

        Parameters
        ----------
        run_fn : function
            the function to run a single trial. This is typically something
            like :meth:`BaseExperiment.run`
        search_algo : :class:`SearchAlgorithm`
            The actual algorithm producing different trials (as a generator)
        gpus : list
            a list of lists. Each list-item in this list specifies a GPU-set
            to use.
            E.g if you want to run one trial on GPU1 and one trial on GPU2,
            this value would be [[1], [2]]; If you would like to run one trial
            on GPU1 and one trial on GPU2 and GPU3 (for example if GPU1 is
            way more powerful), this value would be [[1], [2, 3]].
        **kwargs :
            additional keyword arguments (will be forwarded to ``run_fn``)
        """
        if gpus is None or not gpus:
            gpus = [[]]
        self.search_algo = search_algo
        self._kwargs = kwargs
        self._run_fn = run_fn
        self._gpus = gpus
        self.semaphore = Semaphore(len(gpus))
        self._locks = [Lock() for tmp in gpus]

    @staticmethod
    def start_trial(run_fn, stopping_criterion, trial, gpus, semaphore, locks,
                    kwargs):
        """
        Starts a single trial (waits until a GPU-set is free first)

        Parameters
        ----------
        run_fn : function
        the function to run a single trial. This is typically something
            like :meth:`BaseExperiment.run`
        stopping_criterion : function
            a function to implement, when a trial should stop; This must be
            implemented separately for each scheduler
        trial : :class:`Parameters`
            The trial parameters
        gpus : list
            list of lists, specifying all GPU-sets
        semaphore : :class:`asyncio.Semaphore`
            a semaphore specifying whether there is a free GPU-set
        locks : list
            list of :class:`asyncio.Lock`, specifying for each GPU-set if it
            is currently occupied
        kwargs : dict
            additional keyword arguments, will be forwarded to ``run_fn``

        Returns
        -------
        dict
            metrics from current trial
        :class:`Parameters`
            the current trial's parameters

        """

        # wait until a GPU-Set is free
        async with semaphore:

            # check which GPU-Set is free
            gpu_set_idx = None
            lock_to_use = None
            for idx, lock in enumerate(locks):
                if not lock.locked():
                    gpu_set_idx = idx
                    lock_to_use = lock
                    break

            # Run with specified GPU-Set (locking it, so no other process uses
            # it)
            async with lock_to_use:

                gpus_to_use = gpus[gpu_set_idx]

                reporter_callback = ReporterCallback(stopping_criterion)

                kwargs = {**kwargs}

                if "callbacks" in kwargs:
                    kwargs["callbacks"].append(reporter_callback)
                else:
                    kwargs["callbacks"] = [reporter_callback]

                kwargs.update(gpu_ids=gpus_to_use)

                _ = run_fn(params=trial, **kwargs)
                return {k: v[-1]
                        for k, v in reporter_callback._metrics.items()}, trial

    def __call__(self, **kwargs):
        """
        Executes Trials (in Parallel if possible)

        Parameters
        ----------
        **kwargs :
            additional keyword arguments

        Returns
        -------
        dict
            a dictionary of lists containing all metrics
        tuple
            a tuple containing the Parameters from all trials

        """
        kwargs.update(self._kwargs)

        with DebugSafePool(len(self._gpus)) as p:
            # return metrics and parameters from all trials,
            # search algorithm must be iterable/generator
            metrics_trials = p.map(
                partial(self.start_trial,
                        stopping_criterion=self.stopping_condition,
                        gpus=self._gpus, semaphore=self.semaphore,
                        locks=self._locks, kwargs=kwargs),
                self.search_algo)

            metrics, trials = tuple(zip(*metrics_trials))
            metrics = _concat_dict_items(metrics)

        return metrics, trials

    @abstractmethod
    def stopping_condition(self, reporter) -> bool:
        """
        A stopping condition; must be implemented by each Scheduler

        Parameters
        ----------
        reporter : :class:`ReporterCallback`
            the reporter holding the actual state for the stopping condition
            evaluation

        Returns
        -------
        bool
            whether to stop or not

        Raises
        ------
        NotImplementedError
            If not overwritten by subclass

        """
        raise NotImplementedError


class FIFOScheduler(Scheduler):

    # never stop due to scheduling condition
    def stopping_condition(self, reporter) -> bool:
        return False

