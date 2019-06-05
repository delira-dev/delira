import abc
import logging

from delira import get_backends
import numpy as np

file_logger = logging.getLogger(__name__)


class AbstractNetwork(object):
    """
    Abstract class all networks should be derived from

    """

    _init_kwargs = {}

    @abc.abstractmethod
    def __init__(self, **kwargs):
        """
        Init function to register init kwargs (should be called from all
        subclasses)

        Parameters
        ----------
        **kwargs
            keyword arguments (will be registered to `self.init_kwargs`)

        """
        super().__init__()
        for key, val in kwargs.items():
            self._init_kwargs[key] = val

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """
        AbstractMethod to specify that each model should be able to be called
        for predictions

        Parameters
        ----------
        *args :
            Positional arguments
        **kwargs :
            Keyword Arguments

        Raises
        ------
        NotImplementedError
            if not overwritten by subclass

        """
        raise NotImplementedError()

    @staticmethod
    @abc.abstractmethod
    def closure(model, data_dict: dict, optimizers: dict, losses={},
                metrics={}, fold=0, **kwargs):
        """
        Function which handles prediction from batch, logging, loss calculation
        and optimizer step

        Parameters
        ----------
        model : :class:`AbstractNetwork`
            model to forward data through
        data_dict : dict
            dictionary containing the data
        optimizers : dict
            dictionary containing all optimizers to perform parameter update
        losses : dict
            Functions or classes to calculate losses
        metrics : dict
            Functions or classes to calculate other metrics
        fold : int
            Current Fold in Crossvalidation (default: 0)
        kwargs : dict
            additional keyword arguments

        Returns
        -------
        dict
            Metric values (with same keys as input dict metrics)
        dict
            Loss values (with same keys as input dict losses)
        dict
            Arbitrary number of predictions

        Raises
        ------
        NotImplementedError
            If not overwritten by subclass

        """
        raise NotImplementedError()

    @staticmethod
    def prepare_batch(batch: dict, input_device, output_device):
        """
        Converts a numpy batch of data and labels to suitable datatype and
        pushes them to correct devices

        Parameters
        ----------
        batch : dict
            dictionary containing the batch (must have keys 'data' and 'label'
        input_device :
            device for network inputs
        output_device :
            device for network outputs

        Returns
        -------
        dict
            dictionary containing all necessary data in right format and type
            and on the correct device

        Raises
        ------
        NotImplementedError
            If not overwritten by subclass

        """

        raise NotImplementedError()

    @property
    def init_kwargs(self):
        """
        Returns all arguments registered as init kwargs

        Returns
        -------
        dict
            init kwargs

        """
        return self._init_kwargs


if "SKLEARN" in get_backends():
    from sklearn.base import BaseEstimator
    from inspect import signature as get_signature

    class SklearnEstimator(AbstractNetwork):
        """
        Wrapper Class to wrap all ``sklearn`` estimators and provide delira
        compatibility
        """

        def __init__(self, module: BaseEstimator):
            """

            Parameters
            ----------
            module : :class:`sklearn.base.BaseEstimator`
                the module to wrap
            """

            super().__init__()

            self.module = module

            # forwards methods to self.module if necessary

            for key in ["fit", "partial_fit", "predict"]:
                if hasattr(self.module, key):
                    setattr(self, key, getattr(self.module, key))

            # if estimator is build dynamically based on input, classes have to
            # be passed at least at first time (we pass it every time), because
            # not every class is present in  every batch
            # variable is initialized here, but feeded during the training
            if (self.iterative_training and
                    "classes" in get_signature(self.partial_fit).parameters):
                self.classes = None

        def __call__(self, *args, **kwargs):
            """
            Calls ``self.predict`` with args and kwargs

            Parameters
            ----------
            *args :
                positional arguments of arbitrary number and type
            **kwargs :
                keyword arguments of arbitrary number and type

            Returns
            -------
            dict
                dictionary containing the predictions under key 'pred'

            """
            return {"pred": self.predict(*args, **kwargs)}

        @property
        def iterative_training(self):
            """
            Property indicating, whether a the current module can be
            trained iteratively (batchwise)

            Returns
            -------
            bool
                True: if current module can be trained iteratively
                False: else

            """
            return hasattr(self, "partial_fit")

        @staticmethod
        def prepare_batch(batch: dict, input_device, output_device):
            """
            Helper Function to prepare Network Inputs and Labels (convert them to
            correct type and shape and push them to correct devices)

            Parameters
            ----------
            batch : dict
                dictionary containing all the data
            input_device : Any
                device for module inputs (will be ignored here; just given for
                compatibility)
            output_device : torch.device
                device for module outputs (will be ignored here; just given for
                compatibility)

            Returns
            -------
            dict
                dictionary containing data in correct type and shape and on correct
                device

            """

            new_batch = {"X": batch["data"]}
            if "label" in batch:
                new_batch["y"] = batch["label"].ravel()

            return new_batch

        @staticmethod
        def closure(model, data_dict: dict, optimizers: dict, losses={},
                    metrics={}, fold=0, **kwargs):
            """
            closure method to do a single training step


            Parameters
            ----------
            model : :class:`SkLearnEstimator`
                trainable model
            data_dict : dict
                dictionary containing the data
            optimizers : dict
                dictionary of optimizers to optimize model's parameters;
                ignored here, just passed for compatibility reasons
            losses : dict
                dict holding the losses to calculate errors;
                ignored here, just passed for compatibility reasons
            metrics : dict
                dict holding the metrics to calculate
            fold : int
                Current Fold in Crossvalidation (default: 0)
            **kwargs:
                additional keyword arguments

            Returns
            -------
            dict
                Metric values (with same keys as input dict metrics)
            dict
                Loss values (with same keys as input dict losses; will always
                be empty here)
            list
                Arbitrary number of predictions as torch.Tensor

            """

            if model.iterative_training:
                fit_fn = model.partial_fit

            else:
                fit_fn = model.fit

            if hasattr(model, "classes"):
                # classes must be specified here, because not all classes
                # must be present in each batch and some estimators are build
                # dynamically
                fit_fn(**data_dict, classes=model.classes)
            else:
                fit_fn(**data_dict)

            preds = model(data_dict.pop("X"))

            metric_vals = {}

            for key, metric_fn in metrics.items():
                metric_vals[key] = metric_fn(preds["pred"], **data_dict)

            return metric_vals, {}, preds


if "TORCH" in get_backends():
    import torch

    class AbstractPyTorchNetwork(AbstractNetwork, torch.nn.Module):
        """
        Abstract Class for PyTorch Networks

        See Also
        --------
        `torch.nn.Module`
        :class:`AbstractNetwork`

        """
        @abc.abstractmethod
        def __init__(self, **kwargs):
            """

            Parameters
            ----------
            **kwargs :
                keyword arguments (are passed to :class:`AbstractNetwork`'s `
                __init__ to register them as init kwargs

            """
            torch.nn.Module.__init__(self)
            AbstractNetwork.__init__(self, **kwargs)

        @abc.abstractmethod
        def forward(self, *inputs):
            """
            Forward inputs through module (defines module behavior)
            Parameters
            ----------
            inputs : list
                inputs of arbitrary type and number

            Returns
            -------
            Any
                result: module results of arbitrary type and number

            """
            raise NotImplementedError()

        def __call__(self, *args, **kwargs):
            """
            Calls Forward method

            Parameters
            ----------
            *args :
                positional arguments (passed to `forward`)
            **kwargs :
                keyword arguments (passed to `forward`)

            Returns
            -------
            Any
                result: module results of arbitrary type and number

            """
            return torch.jit.ScriptModule.__call__(self, *args, **kwargs)

        @staticmethod
        def prepare_batch(batch: dict, input_device, output_device):
            """
            Helper Function to prepare Network Inputs and Labels (convert them
            to correct type and shape and push them to correct devices)

            Parameters
            ----------
            batch : dict
                dictionary containing all the data
            input_device : torch.device
                device for network inputs
            output_device : torch.device
                device for network outputs

            Returns
            -------
            dict
                dictionary containing data in correct type and shape and on
                correct device

            """
            return_dict = {"data": torch.from_numpy(batch.pop("data")).to(
                input_device).to(torch.float)}

            for key, vals in batch.items():
                return_dict[key] = torch.from_numpy(vals).to(output_device).to(
                    torch.float)

            return return_dict

    class AbstractTorchScriptNetwork(AbstractNetwork, torch.jit.ScriptModule):

        """
        Abstract Interface Class for TorchScript Networks. For more information
        have a look at https://pytorch.org/docs/stable/jit.html#torchscript

        Warnings
        --------
        In addition to the here defined API, a forward function must be
        implemented and decorated with ``@torch.jit.script_method``

        """
        @abc.abstractmethod
        def __init__(self, optimize=True, **kwargs):
            """

            Parameters
            ----------
            optimize : bool
                whether to optimize the network graph or not; default: True
            **kwargs :
                additional keyword arguments
                (passed to :class:`AbstractNetwork`)
            """
            torch.jit.ScriptModule.__init__(self, optimize=optimize)
            AbstractNetwork.__init__(self, **kwargs)

        def __call__(self, *args, **kwargs):
            """
            Calls Forward method

            Parameters
            ----------
            *args :
                positional arguments (passed to `forward`)
            **kwargs :
                keyword arguments (passed to `forward`)

            Returns
            -------
            Any
                result: module results of arbitrary type and number

            """
            return torch.jit.ScriptModule.__call__(self, *args, **kwargs)

        @staticmethod
        def prepare_batch(batch: dict, input_device, output_device):
            """
            Helper Function to prepare Network Inputs and Labels (convert them
            to correct type and shape and push them to correct devices)

            Parameters
            ----------
            batch : dict
                dictionary containing all the data
            input_device : torch.device
                device for network inputs
            output_device : torch.device
                device for network outputs

            Returns
            -------
            dict
                dictionary containing data in correct type and shape and on
                correct device

            """
            return_dict = {"data": torch.from_numpy(batch.pop("data")).to(
                input_device).to(torch.float)}

            for key, vals in batch.items():
                return_dict[key] = torch.from_numpy(vals).to(output_device).to(
                    torch.float)

            return return_dict


if "TF" in get_backends():
    import tensorflow as tf

    class AbstractTfNetwork(AbstractNetwork, metaclass=abc.ABCMeta):
        """
        Abstract Class for Tf Networks

        See Also
        --------
        :class:`AbstractNetwork`

        """

        @abc.abstractmethod
        def __init__(self, sess=tf.Session, **kwargs):
            """

            Parameters
            ----------
            **kwargs :
                keyword arguments (are passed to :class:`AbstractNetwork`'s `
                __init__ to register them as init kwargs

            """
            AbstractNetwork.__init__(self, **kwargs)
            self._sess = sess()
            self.inputs = {}
            self.outputs_train = {}
            self.outputs_eval = {}
            self._losses = None
            self._optims = None
            self.training = True

        def __call__(self, *args, **kwargs):
            """
            Wrapper for calling self.run in eval setting

            Parameters
            ----------
            *args :
                positional arguments (passed to `self.run`)
            **kwargs:
                keyword arguments (passed to `self.run`)

            Returns
            -------
            Any
                result: module results of arbitrary type and number

            """
            self.training = False
            return self.run(*args, **kwargs)

        def _add_losses(self, losses: dict):
            """
            Add losses to the model graph

            Parameters
            ----------
            losses : dict
                dictionary containing losses.

            """
            raise NotImplementedError()

        def _add_optims(self, optims: dict):
            """
            Add optimizers to the model graph

            Parameters
            ----------
            optims : dict
                dictionary containing losses.
            """
            raise NotImplementedError()

        def run(self, *args, **kwargs):
            """
            Evaluates `self.outputs_train` or `self.outputs_eval` based on
            `self.training`

            Parameters
            ----------
            *args :
                currently unused, exist for compatibility reasons
            **kwargs :
                kwargs used to feed as ``self.inputs``. Same keys as for
                ``self.inputs`` must be used

            Returns
            -------
            dict
                sames keys as outputs_train or outputs_eval,
                containing evaluated expressions as values

            """
            _feed_dict = {}

            for feed_key, feed_value in kwargs.items():
                assert feed_key in self.inputs.keys(), \
                    "{} not found in self.inputs".format(feed_key)
                _feed_dict[self.inputs[feed_key]] = feed_value

            if self.training:
                return self._sess.run(self.outputs_train, feed_dict=_feed_dict)
            else:
                return self._sess.run(self.outputs_eval, feed_dict=_feed_dict)

    class AbstractTfEagerNetwork(AbstractNetwork, tf.keras.layers.Layer):
        def __init__(self, data_format="channels_first", trainable=True,
                     name=None, dtype=None, **kwargs):
            tf.keras.layers.Layer.__init__(self, trainable=trainable,
                                           name=name, dtype=dtype)

            AbstractNetwork.__init__(self, **kwargs)

            self.data_format = data_format
            self.device = "/cpu:0"

        @abc.abstractmethod
        def call(self, *args, **kwargs):
            raise NotImplementedError

        def __call__(self, *args, **kwargs):
            return self.call(*args, **kwargs)

        @staticmethod
        def prepare_batch(batch: dict, input_device, output_device):
            new_batch = {}
            with tf.device(output_device):
                new_batch["label"] = tf.convert_to_tensor(
                    batch["label"].astype(np.float32))

            with tf.device(input_device):
                for k, v in batch.items():
                    if k == "label":
                        continue
                    new_batch[k] = tf.convert_to_tensor(v.astype(np.float32))

            return new_batch

if "CHAINER" in get_backends():
    import chainer
    import numpy as np

    # Use this Mixin Class to set __call__ to None, because there is an
    # internal check inside chainer.Link.__call__ for other __call__ methods
    # of parent classes to be not None. If this would be the case,
    # this function would be executed instead of our forward

    class ChainerMixin(AbstractNetwork):
        __call__ = None

    class AbstractChainerNetwork(chainer.Chain, ChainerMixin):
        """
        Abstract Class for Chainer Networks
        """

        def __init__(self, **kwargs):
            """

            Parameters
            ----------
            **kwargs :
                keyword arguments of arbitrary number and type
                (will be registered as ``init_kwargs``)

            """
            chainer.Chain.__init__(self)
            AbstractNetwork.__init__(self, **kwargs)

        @abc.abstractmethod
        def forward(self, *args, **kwargs) -> dict:
            """
            Feeds Arguments through the network

            Parameters
            ----------
            *args :
                positional arguments of arbitrary number and type
            **kwargs :
                keyword arguments of arbitrary number and type

            Returns
            -------
            dict
                dictionary containing all computation results

            """
            raise NotImplementedError

        def __call__(self, *args, **kwargs) -> dict:
            """
            Makes instances of this class callable.
            Calls the ``forward`` method.

            Parameters
            ----------
            *args :
                positional arguments of arbitrary number and type
            **kwargs :
                keyword arguments of arbitrary number and type

            Returns
            -------
            dict
                dictionary containing all computation results

            """

            return chainer.Chain.__call__(self, *args, **kwargs)

        @staticmethod
        def prepare_batch(batch: dict, input_device, output_device):
            """
            Helper Function to prepare Network Inputs and Labels (convert them
            to correct type and shape and push them to correct devices)

            Parameters
            ----------
            batch : dict
                dictionary containing all the data
            input_device : chainer.backend.Device or string
                device for network inputs
            output_device : torch.device
                device for network outputs

            Returns
            -------
            dict
                dictionary containing data in correct type and shape and on
                correct device

            """
            new_batch = {k: chainer.as_variable(v.astype(np.float32))
                         for k, v in batch.items()}

            for k, v in new_batch.items():
                if k == "data":
                    device = input_device
                else:
                    device = output_device

                # makes modification inplace!
                v.to_device(device)

            return new_batch
