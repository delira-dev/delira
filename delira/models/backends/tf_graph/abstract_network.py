import abc
import tensorflow as tf

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