import logging

import numpy as np
from tqdm import tqdm

from ..data_loading import BaseDataManager

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

    def __init__(self, model, key_mapping: dict, 
                 convert_batch_to_npy_fn=lambda x: x,
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
        convert_batch_to_npy_fn : type, optional
            function converting a batch-tensor to numpy, default: identity 
            function
        prepare_batch_fn : type, optional
            function converting a batch-tensor to the framework specific 
            tensor-type and pushing it to correct device, default: identity 
            function

        """

        self._setup(model, key_mapping, convert_batch_to_npy_fn, 
                    prepare_batch_fn, **kwargs)

        self._tqdm_desc = "Test"

    def _setup(self, network, key_mapping, convert_batch_to_npy_fn, 
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
            a callable function to convert tensors to numpy
        prepare_batch_fn : type
            function converting a batch-tensor to the framework specific
            tensor-type and pushing it to correct device, default: identity
            function

        """
        self.module = network
        self.key_mapping = key_mapping
        self._convert_batch_to_npy_fn = convert_batch_to_npy_fn
        self._prepare_batch = prepare_batch_fn

    def __call__(self, data: dict):
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
        return self.predict(data)

    def predict(self, data: dict):

        data = self._prepare_batch(data)

        mapped_data = {
            k: data[v] for k, v in self.key_mapping.items()}

        pred = self.module(
                **mapped_data
            )

        # converts positional arguments and keyword arguments,
        # but returns only keyword arguments, since positional
        # arguments are not given.
        return self._convert_batch_to_npy_fn(
            **pred
        )[1]

    def predict_data_mgr(self, datamgr, batchsize=None, metrics={},
                         metric_keys=None, verbose=False):
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

        """

        orig_num_aug_processes = datamgr.n_process_augmentation
        orig_batch_size = datamgr.batch_size

        if batchsize is None:
            batchsize = orig_batch_size

        datamgr.batch_size = 1
        datamgr.n_process_augmentation = 1

        batchgen = datamgr.get_batchgen()

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

                preds = self.predict(batch_dict)

                # calculate metrics for predicted batch
                _metric_vals = self.calc_metrics({**batch_dict, **preds},
                                                 metrics=metrics,
                                                 metric_keys=metric_keys)

                for k, v in _metric_vals.items():
                    metric_vals[k].append(v)

                predictions_all.append(preds)

                batch_list = []

        batchgen._finish()

        # convert predictions from list of dicts to dict of lists
        new_predictions_all = {}
        for preds in predictions_all:
            for k, v in preds.items():
                if k in new_predictions_all:
                    new_predictions_all[k].append(v)
                else:
                    new_predictions_all[k] = [v]
                
        # concatenate lists to single arrays
        predictions_all = {k: np.concatenate(_outputs) 
                           for k, _outputs in new_predictions_all.items()}

        for k, v in metric_vals.items():
            metric_vals[k] = np.array(v)

        datamgr.batch_size = orig_batch_size
        datamgr.n_process_augmentation = orig_num_aug_processes

        return predictions_all, metric_vals

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
    def calc_metrics(batch_dict, metrics={}, metric_keys=None):
        """
        Compute metrics

        Parameters
        ----------
        batch_dict : dict
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

        return {key: metric_fn(batch_dict[metric_keys[key][0]], 
                               batch_dict[metric_keys[key][1]])
                for key, metric_fn in metrics.items()}
