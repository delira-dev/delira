from .callbacks import AbstractCallback
from batchgenerators.dataloading import MultiThreadedAugmenter
import pickle
from abc import abstractmethod
import numpy as np
from ..data_loading import BaseDataManager
from tqdm import tqdm
import logging

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

    def __init__(self, model, convert_batch_to_npy_fn=lambda *x: x,
                 prepare_batch_fn=lambda *x: x, **kwargs):
        """

        Parameters
        ----------
        model : :class:`AbstractNetwork`
            the model to predict from
        convert_batch_to_npy_fn : type, optional
            function converting a batch-tensor to numpy, default: identity 
            function
        prepare_batch_fn : [type], optional
            function converting a batch-tensor to the framework specific 
            tensor-type and pushing it to correct device, default: identity 
            function
                    
        """

        self._setup(model, convert_batch_to_npy_fn, prepare_batch_fn, **kwargs)

    def _setup(self, network, convert_batch_to_npy_fn, prepare_batch_fn):
        self.module = network
        self._convert_batch_to_npy_fn = convert_batch_to_npy_fn
        self._prepare_batch = prepare_batch_fn

    def __call__(self, data):
        return self.predict(data)

    def predict(self, data, keys=["data"]):
        data = self._prepare_batch(data)

        if isinstance(data, dict):
            data = [data[key] for key in keys]

        return self._convert_batch_to_npy_fn(
            self.module(
                *data
            )
        )

    def predict_data_mgr(self, datamgr, batchsize=None, verbose=False):
        """
        Defines a routine to predict data obtained from a batchgenerator

        Parameters
        ----------
        batchgen : MultiThreadedAugmenter
            Generator Holding the Batches
        batchsize : Artificial batchsize (sampling will be done with batchsize
                    1 and sampled data will be stacked to match the artificial
                    batchsize)(default: None)

        Raises
        ------
        NotImplementedError
            If not overwritten by subclass

        """

        orig_num_aug_processes = datamgr.n_process_augmentation
        orig_batch_size = datamgr.batch_size

        datamgr.batch_size = 1
        datamgr.n_process_augmentation = 1

        batchgen = datamgr.get_batchgen()

        predictions_all, inputs_all = [], []

        n_batches = batchgen.generator.num_batches * batchgen.num_processes

        if verbose:
            iterable = tqdm(enumerate(batchgen), unit=' sample',
                            total=n_batches, desc='Test')

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

                predictions_all.append(preds)
                inputs_all.append(batch_dict)

                batch_list = []

        batchgen._finish()

        predictions_all = zip(*predictions_all)
        predictions_all = [np.vstack(_outputs) for _outputs in predictions_all]

        input_keys = inputs_all[0].keys()

        total_input_dict = {}

        for key in input_keys:
            total_input_dict[key] = np.concatenate(
                [_input[key] for _input in inputs_all])

        datamgr.batch_size = orig_batch_size
        datamgr.n_process_augmentation = orig_num_aug_processes

        return predictions_all, total_input_dict

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
    def calc_metrics(groundtruths, *predictions, metrics={}, metric_keys=None):
        if metric_keys is None:
            metric_keys = {k: "label" for k in metrics.keys()}

        return {key: metric_fn(groundtruths[metric_keys[key]], *predictions)
                for key, metric_fn in metrics.items()}
