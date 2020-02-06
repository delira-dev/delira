import abc
import logging
import tensorflow as tf
import numpy as np

from delira.models.abstract_network import AbstractNetwork


class AbstractTfGraphNetwork(AbstractNetwork, metaclass=abc.ABCMeta):
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

        return self._sess.run(self.outputs_eval, feed_dict=_feed_dict)

    def _add_losses(self, losses: dict):
        """
        Adds losses to model that are to be used by optimizers or
        during evaluation. Can be overwritten for more advanced loss behavior

        Parameters
        ----------
        losses : dict
            dictionary containing all losses. Individual losses are averaged

        """
        if self._losses is not None and losses:
            logging.warning('Change of losses is not yet supported')
            raise NotImplementedError()

        elif self._losses is not None and not losses:
            pass

        else:
            self._losses = {}
            for name, _loss in losses.items():
                self._losses[name] = _loss(self.inputs["label"],
                                           self.outputs_train["pred"])

            total_loss = tf.reduce_mean(list(self._losses.values()), axis=0)

            self._losses['total'] = total_loss
            self.outputs_train["losses"] = self._losses
            self.outputs_eval["losses"] = self._losses

    def _add_optims(self, optims: dict):
        """
        Adds optims to model that are to be used by optimizers or during
        training. Can be overwritten for more advanced optimizers

        Parameters
        ----------
        optim: dict
            dictionary containing all optimizers, optimizers should be of
            Type[tf.train.Optimizer]

        """
        if self._optims is not None and optims:
            logging.warning('Change of optims is not yet supported')
        elif self._optims is not None and not optims:
            pass
        else:
            self._optims = optims['default']
            grads = self._optims.compute_gradients(self._losses['total'])
            step = self._optims.apply_gradients(grads)
            self.outputs_train["default_step"] = step

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
        output_device : Any
            device for module outputs (will be ignored here; just given for
            compatibility)

        Returns
        -------
        dict
            dictionary containing data in correct type and shape and on correct
            device

        """
        return {k: v.astype(np.float32) for k, v in batch.items()}

    @staticmethod
    def closure(model, data_dict: dict, optimizers: dict, losses: dict,
                iter_num: int, fold=0, **kwargs):
        """
        default closure method to do a single training step;
        Could be overwritten for more advanced models

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
        iter_num: int
            the number of of the current iteration in the current epoch;
            Will be restarted at zero at the beginning of every epoch
        fold : int
            Current Fold in Crossvalidation (default: 0)
        **kwargs:
            additional keyword arguments

        Returns
        -------
        dict
            Loss values (with same keys as input dict losses; will always
            be empty here)
        dict
            dictionary containing all predictions

        """

        inputs = data_dict['data']

        outputs = model.run(data=inputs, label=data_dict['label'])
        loss_vals = outputs['losses']

        return loss_vals, outputs
