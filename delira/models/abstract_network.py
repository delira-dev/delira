import abc
import logging
from delira import get_backends

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
    def closure(model, data_dict: dict, optimizers: dict, criterions={},
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
        criterions : dict
            Functions or classes to calculate criterions
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
            Loss values (with same keys as input dict criterions)
        list
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
            return torch.nn.Module.__call__(self, *args, **kwargs)

        @staticmethod
        def prepare_batch(batch: dict, input_device, output_device):
            """
            Helper Function to prepare Network Inputs and Labels (convert them to
            correct type and shape and push them to correct devices)

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
                dictionary containing data in correct type and shape and on correct
                device

            """
            return_dict = {"data": torch.from_numpy(batch.pop("data")).to(
                input_device).to(torch.float)}

            for key, vals in batch.items():
                return_dict[key] = torch.from_numpy(vals).to(output_device).to(
                    torch.float)

            return return_dict

if "TF" in get_backends():
    import tensorflow as tf

    class AbstractTfNetwork(AbstractNetwork):
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
            self.inputs = None
            self.outputs_train = None
            self.outputs_eval = None
            self._losses = None
            self._optims = None
            self.training = True

        def __call__(self, *args):
            """
            Wrapper for calling self.run in eval setting

            Parameters
            ----------
            *args :
                positional arguments (passed to `self.run`)

            Returns
            -------
            Any
                result: module results of arbitrary type and number

            """
            self.training = False
            return self.run(*args)

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

        def run(self, *args):
            """
            Evaluates `self.outputs_train` or `self.outputs_eval` based on `self.training`

            Parameters
            ----------
            *args :
                arguments to feed as `self.inputs`. Must have same length as `self.inputs`

            Returns
            -------
            np.ndarray or list
                based on len(self.outputs*), returns either list or np.ndarray

            """
            if isinstance(self.inputs, tf.Tensor):
                _feed_dict = dict(zip([self.inputs], args))
            else:
                _feed_dict = dict(zip(self.inputs, args))
            if self.training:
                return self._sess.run(self.outputs_train, feed_dict=_feed_dict)
            else:
                return self._sess.run(self.outputs_eval, feed_dict=_feed_dict)
