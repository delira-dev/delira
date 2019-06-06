import logging

import numpy as np
from tqdm import tqdm

from delira.data_loading import BaseDataManager
from delira.training.utils import convert_to_numpy_identity
from delira.utils.config import LookupConfig

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
            convert_batch_to_npy_fn=convert_to_numpy_identity,
            prepare_batch_fn=lambda **x: x, **kwargs):
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
        prepare_batch_fn : (dict, str, str) -> dict
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

    def predict_data_mgr(self, datamgr, batchsize=None, metrics={},
                         metric_keys=None, verbose=False, lazy_gen=False,
                         **kwargs):
        """
        Defines a routine to predict data obtained from a batchgenerator

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
        lazy_gen : bool
            if True: Yields results instead of returning them; should be
            specified if predicting on a low-memory device or when results
            should be saved immediately
        kwargs :
            keyword arguments passed to :func:`prepare_batch_fn`

        Yields
        ------
        dict
            a dictionary containing all predictions of the current batch
            if ``lazy_gen`` is True
        dict
            a dictionary containing all metrics of the current batch
            if ``lazy_gen`` is True
        dict
            a dictionary containing all predictions;
            if ``lazy_gen`` is False
        dict
            a dictionary containing all validation metrics (maybe empty);
            if ``lazy_gen`` is False

        """

        orig_num_aug_processes = datamgr.n_process_augmentation
        orig_batch_size = datamgr.batch_size

        if batchsize is None:
            batchsize = orig_batch_size

        datamgr.batch_size = 1
        datamgr.n_process_augmentation = 1

        batchgen = datamgr.get_batchgen()

        if not lazy_gen:
            predictions_all, metric_vals = [], {k: [] for k in metrics.keys()}

        n_batches = batchgen.generator.num_batches * batchgen.num_processes

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
                for batch in batch_list:
                    for key, val in batch.items():
                        if key in batch_dict.keys():
                            batch_dict[key].append(val)
                        else:
                            batch_dict[key] = [val]

                for key, val_list in batch_dict.items():
                    batch_dict[key] = np.concatenate(val_list)

                preds = self.predict(batch_dict, **kwargs)

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

                if lazy_gen:
                    yield preds, _metric_vals
                else:
                    for k, v in _metric_vals.items():
                        metric_vals[k].append(v)

                    predictions_all.append(preds)

                batch_list = []

        batchgen._finish()
        datamgr.batch_size = orig_batch_size
        datamgr.n_process_augmentation = orig_num_aug_processes

        if lazy_gen:
            # triggers stopiteration
            return

        # convert predictions from list of dicts to dict of lists
        new_predictions_all = {}

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

                    new_dict[k] = __convert_dict(v, new_dict[k])

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
                        new_predictions_all[k].append(v)
                    else:
                        new_predictions_all[k] = [v]

            return new_dict

        # recursively convert all nested dicts
        for preds in predictions_all:
            new_predictions_all = __convert_dict(preds, new_predictions_all)

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
                    v = __concatenate_dict_items(v)
                else:
                    v = np.concatenate(v)

                dict_like[k] = v

                return dict_like

        # concatenate lists to single arrays
        predictions_all = __concatenate_dict_items(new_predictions_all)

        for k, v in metric_vals.items():
            metric_vals[k] = np.array(v)

        # must yield these instead of returning them,
        # because every function with a yield in it's body returns a
        # generator object (even if the yield is never triggered)
        yield predictions_all, metric_vals

        # triggers stopiteration
        return

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
    def calc_metrics(batch: LookupConfig, metrics={}, metric_keys=None):
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
        if metric_keys is None:
            metric_keys = {k: ("pred", "label") for k in metrics.keys()}

        return {key: metric_fn(*[batch.nested_get(k)
                                 for k in metric_keys[key]])
                for key, metric_fn in metrics.items()}
