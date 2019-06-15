import logging
import copy

import numpy as np
from tqdm import tqdm

from ..data_loading import BaseDataManager
from .train_utils import convert_batch_to_numpy_identity
from ..utils.config import LookupConfig

logger = logging.getLogger(__name__)


class Predictor(object):
    """
    Defines an API for Predictions from a Network

    See Also
    --------
    :class:`PyTorchNetworkTrainer`

    """

    # static variable to prevent certain attributers from overwriting
    __KEYS_TO_GUARD = []

    def __init__(
            self, model, key_mapping: dict,
            convert_batch_to_npy_fn=convert_batch_to_numpy_identity,
            prepare_batch_fn=lambda x: x, **kwargs):
        """

        Parameters
        ----------
        model : :class:`AbstractNetwork`
            the model to predict from
        key_mapping : dict
            a dictionary containing the mapping from the ``data_dict`` to
            the actual model's inputs.
            E.g. if a model accepts one input named 'x' and the data_dict
            contains one entry named 'data' this argument would have to
            be ``{'x': 'data'}``
        convert_batch_args_kwargs_to_npy_fn : type, optional
            a callable function to convert tensors in positional and keyword
            arguments to numpy; default: identity function
        prepare_batch_fn : type, optional
            function converting a batch-tensor to the framework specific
            tensor-type and pushing it to correct device, default: identity
            function
        **kwargs :
            additional keyword arguments

        """

        self._setup(model, key_mapping, convert_batch_to_npy_fn,
                    prepare_batch_fn, **kwargs)

        self._tqdm_desc = "Test"

    def _setup(self, network, key_mapping, convert_batch_args_kwargs_to_npy_fn,
               prepare_batch_fn, **kwargs):
        """

        Parameters
        ----------
        network : :class:`AbstractNetwork`
            the network to predict from
        key_mapping : dict
            a dictionary containing the mapping from the ``data_dict`` to
            the actual model's inputs.
            E.g. if a model accepts one input named 'x' and the data_dict
            contains one entry named 'data' this argument would have to
            be ``{'x': 'data'}``
        convert_batch_to_npy_fn : type
            a callable function to convert tensors in positional and keyword
            arguments to numpy
        prepare_batch_fn : type
            function converting a batch-tensor to the framework specific
            tensor-type and pushing it to correct device, default: identity
            function

        """
        self.module = network
        self.key_mapping = key_mapping
        self._convert_to_npy_fn = convert_batch_args_kwargs_to_npy_fn
        self._prepare_batch = prepare_batch_fn

    def __call__(self, data: dict, **kwargs):
        """
        Method to call the class.
        Returns the predictions corresponding to the given data
        obtained by the model

        Parameters
        ----------
        data : dict
            batch dictionary

        Returns
        -------
        dict
            predicted data
        """
        return self.predict(data, **kwargs)

    def predict(self, data: dict, **kwargs):
        """
        Predict single batch
        Returns the predictions corresponding to the given data
        obtained by the model

        Parameters
        ----------
        data : dict
            batch dictionary
        **kwargs :
            keyword arguments(directly passed to ``prepare_batch``)

        Returns
        -------
        dict
            predicted data

        """
        data = self._prepare_batch(data, **kwargs)

        mapped_data = {
            k: data[v] for k, v in self.key_mapping.items()}

        pred = self.module(
            **mapped_data
        )

        # converts positional arguments and keyword arguments,
        # but returns only keyword arguments, since positional
        # arguments are not given.
        return self._convert_to_npy_fn(
            **pred
        )[1]

    def predict_data_mgr(self, datamgr, batchsize=None, metrics=None,
                         metric_keys=None, verbose=False, **kwargs):
        """
        Defines a routine to predict data obtained from a batchgenerator
        without explicitly caching anything

        Parameters
        ----------
        datamgr : :class:`BaseDataManager`
            Manager producing a generator holding the batches
        batchsize : int
            Artificial batchsize (sampling will be done with batchsize
            1 and sampled data will be stacked to match the artificial
            batchsize)(default: None)
        metrics : dict
            the metrics to calculate
        metric_keys : dict
            the ``batch_dict`` items to use for metric calculation
        verbose : bool
            whether to show a progress-bar or not, default: False
        kwargs :
            keyword arguments passed to :func:`prepare_batch_fn`

        Yields
        ------
        dict
            a dictionary containing all predictions of the current batch
        dict
            a dictionary containing all metrics of the current batch

        """
        if metrics is None:
            metrics = {}
        orig_num_aug_processes = datamgr.n_process_augmentation
        orig_batch_size = datamgr.batch_size

        if batchsize is None:
            batchsize = orig_batch_size

        datamgr.batch_size = 1
        datamgr.n_process_augmentation = 1

        batchgen = datamgr.get_batchgen()

        n_batches = batchgen.num_batches

        if verbose:
            iterable = tqdm(enumerate(batchgen), unit=' sample',
                            total=n_batches, desc=self._tqdm_desc)

        else:
            iterable = enumerate(batchgen)

        batch_list = []

        for i, batch in iterable:

            if not batch_list and (n_batches - i) < batchsize:
                batchsize = n_batches - i
                logger.debug("Set Batchsize down to %d to avoid cutting "
                             "of the last batches" % batchsize)

            batch_list.append(batch)

            # if queue is full process queue:
            if batchsize is None or len(batch_list) >= batchsize:

                batch_dict = {}
                for _batch in batch_list:
                    for key, val in _batch.items():
                        if key in batch_dict.keys():
                            batch_dict[key].append(val)
                        else:
                            batch_dict[key] = [val]

                for key, val_list in batch_dict.items():
                    batch_dict[key] = np.concatenate(val_list)

                preds = self.predict(copy.copy(batch_dict), **kwargs)

                # convert batchdict back to numpy (self.predict may convert it
                # to backend-specific tensor type) - no-op if already numpy
                batch_dict = self._convert_to_npy_fn(**batch_dict)[1]

                preds_batch = LookupConfig()
                preds_batch.update(batch_dict)
                preds_batch.update(preds)

                # calculate metrics for predicted batch
                _metric_vals = self.calc_metrics(preds_batch,
                                                 metrics=metrics,
                                                 metric_keys=metric_keys)

                yield preds, _metric_vals

                batch_list = []

        batchgen._finish()
        datamgr.batch_size = orig_batch_size
        datamgr.n_process_augmentation = orig_num_aug_processes

        return

    def predict_data_mgr_cache_metrics_only(self, datamgr, batchsize=None,
                                            metrics=None, metric_keys=None,
                                            verbose=False, **kwargs):
        """
        Defines a routine to predict data obtained from a batchgenerator and
        caches the metrics

        Parameters
        ----------
        datamgr : :class:`BaseDataManager`
            Manager producing a generator holding the batches
        batchsize : int
            Artificial batchsize (sampling will be done with batchsize
            1 and sampled data will be stacked to match the artificial
            batchsize)(default: None)
        metrics : dict
            the metrics to calculate
        metric_keys : dict
            the ``batch_dict`` items to use for metric calculation
        verbose : bool
            whether to show a progress-bar or not, default: False
        kwargs :
            keyword arguments passed to :func:`prepare_batch_fn`

        Yields
        ------
        dict
            a dictionary containing all validation metrics (maybe empty)

        Notes
        -----
        This function stores each prediction temporarily for metric
        calculation; This results in a (typically) way lower memory
        consumption than :meth:`Predictor.predict_data_mgr_cache_all`,
        but still caches the metrics. If this is not desired, it is recommended
        to use :meth:`Predictor.predict_data_mgr` and iterate over the
        generator as this only produces per-batch metrics and predictions and
        does not cache anything by default

        """
        if metrics is None:
            metrics = {}
        yield from self.predict_data_mgr_cache(datamgr=datamgr,
                                               batchsize=batchsize,
                                               metrics=metrics,
                                               metric_keys=metric_keys,
                                               verbose=verbose,
                                               cache_preds=False, **kwargs)

        return

    def predict_data_mgr_cache_all(self, datamgr, batchsize=None, metrics=None,
                                   metric_keys=None, verbose=False, **kwargs):
        """
        Defines a routine to predict data obtained from a batchgenerator and
        caches all predictions and metrics (yields them in dicts)

        Parameters
        ----------
        datamgr : :class:`BaseDataManager`
            Manager producing a generator holding the batches
        batchsize : int
            Artificial batchsize (sampling will be done with batchsize
            1 and sampled data will be stacked to match the artificial
            batchsize)(default: None)
        metrics : dict
            the metrics to calculate
        metric_keys : dict
            the ``batch_dict`` items to use for metric calculation
        verbose : bool
            whether to show a progress-bar or not, default: False
        kwargs :
            keyword arguments passed to :func:`prepare_batch_fn`

        Yields
        ------
        dict
            a dictionary containing all predictions;
        dict
            a dictionary containing all validation metrics (maybe empty)

        Warnings
        --------
        Since this function caches all predictions and metrics, this may result
        in huge memory consumption. If you are running out of memory, please
        have a look at :meth:`Predictor.predict_data_mgr_cache_metrics_only`
        or :meth:`Predictor.predict_data_mgr`

        """
        if metrics is None:
            metrics = {}
        yield from self.predict_data_mgr_cache(datamgr=datamgr,
                                               batchsize=batchsize,
                                               metrics=metrics,
                                               metric_keys=metric_keys,
                                               verbose=verbose,
                                               cache_preds=True, **kwargs)

        return

    def predict_data_mgr_cache(self, datamgr, batchsize=None, metrics=None,
                               metric_keys=None, verbose=False,
                               cache_preds=False, **kwargs):
        """
        Defines a routine to predict data obtained from a batchgenerator and
        caches all predictions and metrics (yields them in dicts)

        Parameters
        ----------
        datamgr : :class:`BaseDataManager`
            Manager producing a generator holding the batches
        batchsize : int
            Artificial batchsize (sampling will be done with batchsize
            1 and sampled data will be stacked to match the artificial
            batchsize)(default: None)
        metrics : dict
            the metrics to calculate
        metric_keys : dict
            the ``batch_dict`` items to use for metric calculation
        verbose : bool
            whether to show a progress-bar or not, default: False
        cache_preds : bool
            whether to also cache predictions
        kwargs :
            keyword arguments passed to :func:`prepare_batch_fn`

        Yields
        ------
        dict
            a dictionary containing all validation metrics (maybe empty)
        dict
            a dictionary containing all predictions; If ``cache_preds=True``

        Warnings
        --------
        Since this function caches all metrics and may additionally cache all
        predictions (based on the argument ``cache_preds``), this may result
        in huge memory consumption. If you are running out of memory, please
        have a look at :meth:`Predictor.predict_data_mgr_cache_metrics_only`
        or :meth:`Predictor.predict_data_mgr` or consider setting
        ``cache_preds`` to ``False`` (if not done already)

        """

        if metrics is None:
            metrics = {}

        predictions_all, metric_vals = [], {k: [] for k in metrics.keys()}

        for preds, _metric_vals in self.predict_data_mgr(
                datamgr=datamgr,
                batchsize=batchsize,
                metrics=metrics,
                metric_keys=metric_keys,
                verbose=verbose,
                **kwargs):

            if cache_preds:
                predictions_all.append(preds)
            for k, v in _metric_vals.items():
                metric_vals[k].append(v)

        if cache_preds:
            # convert predictions from list of dicts to dict of lists
            new_predictions_all = {}

            # recursively convert all nested dicts
            for preds in predictions_all:
                new_predictions_all = self.__convert_dict(preds,
                                                          new_predictions_all)

            # concatenate lists to single arrays
            preds_all = self.__concatenate_dict_items(new_predictions_all)
        else:
            preds_all = {}

        for k, v in metric_vals.items():
            metric_vals[k] = np.array(v)

        if cache_preds:
            yield preds_all, metric_vals
        else:
            yield metric_vals

        return

    @staticmethod
    def __convert_dict(old_dict, new_dict):
        """
        Function to recursively convert dicts

        Parameters
        ----------
        old_dict : dict
            the old nested dict
        new_dict : dict
            the new nested dict

        Returns
        -------
        dict
            the updated new nested dict
        """
        for k, v in old_dict.items():

            # apply same function again on item if item is dict
            if isinstance(v, dict):
                if k not in new_dict:
                    new_dict[k] = {}

                new_dict[k] = Predictor.__convert_dict(v, new_dict[k])

            else:

                # check if v is scalar and convert to npy-array if
                # necessary.
                # Otherwise concatenation might fail
                if np.isscalar(v):
                    v = np.array(v)

                # check for zero-sized arrays and reshape if necessary.
                # Otherwise concatenation might fail
                if v.shape == ():
                    v = v.reshape(1)
                if k in new_dict:
                    new_dict[k].append(v)
                else:
                    new_dict[k] = [v]

        return new_dict

    @staticmethod
    def __concatenate_dict_items(dict_like: dict):
        """
        Function to recursively concatenate dict-items

        Parameters
        ----------
        dict_like : dict
            the (nested) dict, whoose items should be concatenated

        Returns
        -------

        """
        for k, v in dict_like.items():
            if isinstance(v, dict):
                v = Predictor.__concatenate_dict_items(v)
            else:
                v = np.concatenate(v)

            dict_like[k] = v

            return dict_like

    def __setattr__(self, key, value):
        """
        Set attributes and guard specific attributes after they have been set
        once

        Parameters
        ----------
        key : str
            the attributes name
        value : Any
            the value to set

        Raises
        ------
        PermissionError
            If attribute which should be set is guarded

        """

        # check if key has been set once
        if key in self.__KEYS_TO_GUARD and hasattr(self, key):
            raise PermissionError("%s should not be overwritten after "
                                  "it has been set once" % key)
        else:
            super().__setattr__(key, value)

    @staticmethod
    def calc_metrics(batch: LookupConfig, metrics=None, metric_keys=None):
        """
        Compute metrics

        Parameters
        ----------
        batch: LookupConfig
            dictionary containing the whole batch
            (including predictions)
        metrics: dict
            dict with metrics
        metric_keys : dict
            dict of tuples which contains hashables for specifying the items
            to use for calculating the respective metric.
            If not specified for a metric, the keys "pred" and "label"
            are used per default

        Returns
        -------
        dict
            dict with metric results
        """
        if metrics is None:
            metrics = {}
        if metric_keys is None:
            metric_keys = {k: ("pred", "label") for k in metrics.keys()}

        return {key: metric_fn(*[batch.nested_get(k)
                                 for k in metric_keys[key]])
                for key, metric_fn in metrics.items()}
