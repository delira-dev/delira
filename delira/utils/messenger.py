import logging
import warnings
from abc import ABC, abstractmethod

from delira.training import BaseExperiment
from delira.training.callbacks import AbstractCallback


class BaseMessenger(ABC):
    """
    Wrap arbitrary experiments and connect its functions to a
    notification service.
    """

    def __init__(self, experiment: BaseExperiment, notify_epochs: int = None):
        """

        Parameters
        ----------
        experiment : :class:`BaseExperiment`
            instance of current experiment
        notify_epochs : int
            Activates notifications about finished epochs with frequency
            `notify_epochs`.
        """
        super().__init__()
        self._experiment = experiment
        self._notify_epochs = notify_epochs

    @abstractmethod
    def emit_message(self, msg: str) -> dict:
        """
        Emit message.
        Implement this method in base class to create new notification
        services.

        Parameters
        ----------
        msg : str
            message which should be emitted

        Returns
        -------
        dict
            dict with additional information from message
        """
        raise NotImplementedError

    def __getattr__(self, attr):
        """
        If wrapper does not implement attribute, return attribute of wrapped
        object

        Parameters
        ----------
        attr : str
            name of attribute

        Returns
        -------
        Any
            attribute
        """
        # NOTE do note use hasattr, it goes into infinite recursion
        if attr in self.__dict__:
            # this object has it
            return getattr(self, attr)
        return getattr(self._experiment, attr)

    def run(self, *args, **kwargs):
        """
        Wrapper for run function. Notifies experiment start, fail, complete.

        Parameters
        ----------
        args :
            positional arguments passed to experiment.
        kwargs :
            additional keyword arguments passed to experiment.

        Returns
        -------
        Any
            result of experiment
        """
        if self._notify_epochs is not None:
            callbacks = kwargs.pop("callbacks", [])
            callbacks.append(MessengerEpochCallback(self._notify_epochs,
                                                    self))
            kwargs["callbacks"] = callbacks

        msg = str(self._experiment.name) + " : Training started."
        self.emit_message(msg)

        try:
            out = self._experiment.run(*args, **kwargs)
        except Exception as e:
            msg = \
                str(self._experiment.name) + " : Training failed. \n" + str(e)
            self.emit_message(msg)
            raise

        msg = str(self._experiment.name) + " : Training completed."
        self.emit_message(msg)
        return out

    def resume(self, *args, **kwargs):
        """
        Wrapper for resume function. Notifies experiment start, fail, complete.

        Parameters
        ----------
        args :
            positional arguments passed to experiment.
        kwargs :
            additional keyword arguments passed to experiment.

        Returns
        -------
        Any
            result of experiment
        """
        if self._notify_epochs is not None:
            callbacks = kwargs.pop("callbacks", [])
            callbacks.append(MessengerEpochCallback(self._notify_epochs,
                                                    self))
            kwargs["callbacks"] = callbacks

        msg = str(self._experiment.name) + " : Resume started."
        self.emit_message(msg)

        try:
            out = self._experiment.resume(*args, **kwargs)
        except Exception as e:
            msg = str(self._experiment.name) + " : Resume failed. \n" + str(e)
            self.emit_message(msg)
            raise e

        msg = str(self._experiment.name) + " : Resume ended."
        self.emit_message(msg)
        return out

    def test(self, *args, **kwargs):
        """
        Wrapper for test function. Notifies experiment start, fail, complete.

        Parameters
        ----------
        args :
            positional arguments passed to experiment.
        kwargs :
            additional keyword arguments passed to experiment.

        Returns
        -------
        Any
            result of experiment
        """
        msg = str(self._experiment.name) + " : Test started."
        self.emit_message(msg)

        try:
            out = self._experiment.test(*args, **kwargs)
        except Exception as e:
            msg = str(self._experiment.name) + " : Test failed. \n" + str(e)
            self.emit_message(msg)
            raise e

        msg = str(self._experiment.name) + " : Test completed."
        self.emit_message(msg)
        return out

    def kfold(self, *args, **kwargs):
        """
        Wrapper for kfold function. Notifies experiment start, fail, complete,
        end of fold.

        Parameters
        ----------
        args :
            positional arguments passed to experiment.
        kwargs :
            additional keyword arguments passed to experiment.

        Returns
        -------
        Any
            result of experiment
        """
        # append own callback for fold messages
        callbacks = kwargs.pop("callbacks", [])
        callbacks.append(MessengerFoldCallback(self))

        # append own callback for epoch messages
        if self._notify_epochs is not None:
            callbacks.append(MessengerEpochCallback(self._notify_epochs,
                                                    self))

        kwargs["callbacks"] = callbacks

        msg = str(self._experiment.name) + " : Kfold started."
        self.emit_message(msg)

        # execute k-fold
        try:
            out = self._experiment.kfold(*args, **kwargs)
        except Exception as e:
            msg = str(self._experiment.name) + " : Kfold failed. \n" + str(e)
            self.emit_message(msg)
            raise e

        msg = str(self._experiment.name) + " : Kfold completed."
        self.emit_message(msg)

        return out


class MessengerEpochCallback(AbstractCallback):
    """
    Callback for "Epoch X trained" message

    See Also
    --------
    :class:`BaseMessenger`
    """

    def __init__(self, n_epochs: int, messenger: BaseMessenger):
        """

        Parameters
        ----------
        n_epochs : int
            notification frequency
        messenger : :class:`BaseMessenger`
            instance of a experiment with messanger to emit message
        """
        super().__init__()
        self._n_epochs = n_epochs
        self._messenger = messenger

    def at_epoch_end(self, trainer, **kwargs) -> dict:
        """
        Call at end of epoch

        Parameters
        ----------
        trainer : :class:`BaseTrainer`
            instance of trainer
        kwargs :
            additional keyword arguments. Must contain ``curr_epoch``.

        Returns
        -------
        dict
            empty dict
        """
        curr_epoch = kwargs.pop("curr_epoch")
        trained_epochs = curr_epoch - trainer.start_epoch
        if trained_epochs % self._n_epochs == 0:
            msg = "Epoch " + str(curr_epoch) + " trained."
            self._messenger.emit_message(msg)
        return {}


class MessengerFoldCallback(AbstractCallback):
    """
    Callback for "Fold X completed" in slack

    See Also
    --------
    :class:`BaseMessenger`
    """

    def __init__(self, messenger: BaseMessenger):
        """

        Parameters
        ----------
        messenger : :class:`BaseMessenger`
            instance of a experiment with messanger to emit message
        """
        super().__init__()
        self._messenger = messenger

    def at_training_start(self, trainer, **kwargs) -> dict:
        """
        End of training callback

        Parameters
        ----------
        trainer : :class:`BaseTrainer`
            instance of trainer
        kwargs :
            additional keyword arguments (not used)

        Returns
        -------
        dict
            empty dict
        """
        msg = "Fold " + str(trainer.fold) + " started."
        self._messenger.emit_message(msg)
        return {}

    def at_training_end(self, trainer, **kwargs) -> dict:
        """
        End of training callback

        Parameters
        ----------
        trainer : :class:`BaseTrainer`
            instance of trainer
        kwargs :
            additional keyword arguments (not used)

        Returns
        -------
        dict
            empty dict
        """
        msg = "Fold " + str(trainer.fold) + " completed."
        self._messenger.emit_message(msg)
        return {}


class SlackMessenger(BaseMessenger):
    """
    Wrap arbitrary experiments and connect its functions to slack
    notification

    .. note:: `token`can be either your personal user token or a token
              from an artifical bot. To create your own bot you can
              visit https://api.slack.com/ and click 'Your Apps' at the
              top-right corner (you may need to create an own workspace
              where you can install your bot).

    .. warning:: Slack messenger has ´slackclient==1.3.1´ as a dependency which
                 is not included in the requirements!
    """

    def __init__(self, experiment: BaseExperiment, token: str,
                 channel: str, notify_epochs: int = None, **kwargs):
        """

        Parameters
        ----------
        experiment : :class:`BaseExperiment`
            instance of current experiment
        token : str
            User or Bot token from slack
        channel : str
            channel id (destination of messages)
        notify_epochs : int
            Activates notifications about finished epochs with frequency
            `notify_epochs`.
        kwargs :
            additional keyword arguments passed to :class:`SlackClient`

        Raises
        ------
        ImportError
            if `slackclient` is not installed

        See Also
        --------
        :class:`BaseMessenger`
        """
        super().__init__(experiment, notify_epochs=notify_epochs)

        # switch between different versions (with changed imports)
        try:
            from slackclient import SlackClient
        except ImportError as e:
            try:
                from slack import WebClient as SlackClient
            except ImportError as e:

                warnings.warn(
                    "Could not import `slackclient`. This package is not"
                    "included in the default depencies of delira!")
                raise e

        self._client = SlackClient(token, **kwargs)
        self._channel = channel

        # initial slack message
        msg = "Created new experiment: " + str(self._experiment.name)
        resp = self._client.api_call("chat.postMessage",
                                     channel=self._channel,
                                     text=msg)
        if not resp["ok"]:
            logging.error(
                "Slack message was not emitted correctly! \n {}".format(msg))
        self._ts = resp['ts'] if 'ts' in resp else None

    def emit_message(self, msg) -> dict:
        """
        Emit message to current experiment thread

        Parameters
        ----------
        msg : str
            message which should be emitted

        Returns
        -------
        dict
            dict with additional information from message
        """
        resp = self._client.api_call(
            "chat.postMessage",
            channel=self._channel,
            text=msg,
            thread_ts=self._ts,
        )
        if not resp["ok"]:
            logging.error(
                "Slack message was not emitted correctly! \n {}".format(msg))
        return resp
