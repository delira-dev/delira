import logging
from slackclient import SlackClient

from delira.training import BaseExperiment
from delira.training.callbacks import AbstractCallback


class SlackEpochsCallback(AbstractCallback):
    def __init__(self, n_epochs: int, slack_exp):
        """
        Callback for "Epoch X trained" in slack

        Parameters
        ----------
        n_epochs : int
            notification frequency
        slack_exp : :class:`SlackExperiment`
            instance of a slack experiment to emit slack message
        """
        super().__init__()
        self._n_epochs = n_epochs
        self._slack_exp = slack_exp

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
            self._slack_exp.emit_message(msg)
        return {}


class SlackFoldCallback(AbstractCallback):
    def __init__(self, slack_exp):
        """
        Callback for "Fold X completed" in slack

        Parameters
        ----------
        slack_exp : :class:`SlackExperiment`
            instance of a slack experiment to emit slack message
        """
        super().__init__()
        self._slack_exp = slack_exp

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
        self._slack_exp.emit_message(msg)
        return {}


class SlackExperiment(object):
    def __init__(self, experiment: BaseExperiment, token: str,
                 channel: str, *args, notify_epochs=None, **kwargs):
        """
        Wrap arbitrary experiments and connect its functions to slack
        notification

        Parameters
        ----------
        experiment : :class:`BaseExperiment`
            instance of current experiment
        token : str
            User or Bot token from slack
        channel : str
            channel id (destination of messages)
        args :
            additional positional arguments passed to :class:`SlackClient`
        notify_epochs : int
            Activates additional notifications with frequency `notify_epochs`.
        kwargs :
            additional keyword arguments passed to :class:`SlackClient`
        """
        # wrap the object
        self._client = SlackClient(token, *args, **kwargs)
        self._channel = channel
        self._wrapped_exp = experiment
        self._notify_epochs = notify_epochs

        # initial slack message
        msg = "Created new experiment: " + str(self._wrapped_exp.name)
        resp = self._client.api_call("chat.postMessage",
                                     channel=self._channel,
                                     text=msg)
        if not resp["ok"]:
            logging.error(
                "Slack message was not emitted correctly! \n {}".format(msg))
        self._ts = resp['ts']

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
        # see if this object has attr
        # NOTE do not use hasattr, it goes into
        # infinite recurrsion
        if attr in self.__dict__:
            # this object has it
            return getattr(self, attr)
        # proxy to the wrapped object
        return getattr(self._wrapped_exp, attr)

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
            callbacks.append(SlackEpochsCallback(self._notify_epochs,
                                                 self))
            kwargs["callbacks"] = callbacks

        msg = str(self._wrapped_exp.name) + " : Training started."
        self.emit_message(msg)

        try:
            out = self._wrapped_exp.run(*args, **kwargs)
        except Exception as e:
            msg = \
                str(self._wrapped_exp.name) + " : Training failed. \n" + str(e)
            self.emit_message(msg)
            raise e

        msg = str(self._wrapped_exp.name) + " : Training completed."
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
            callbacks.append(SlackEpochsCallback(self._notify_epochs,
                                                 self))
            kwargs["callbacks"] = callbacks

        msg = str(self._wrapped_exp.name) + " : Resume started."
        self.emit_message(msg)

        try:
            out = self._wrapped_exp.resume(*args, **kwargs)
        except Exception as e:
            msg = str(self._wrapped_exp.name) + " : Resume failed. \n" + str(e)
            self.emit_message(msg)
            raise e

        msg = str(self._wrapped_exp.name) + " : Resume ended."
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
        msg = str(self._wrapped_exp.name) + " : Test completed."
        self.emit_message(msg)

        try:
            out = self._wrapped_exp.test(*args, **kwargs)
        except Exception as e:
            msg = str(self._wrapped_exp.name) + " : Test failed. \n" + str(e)
            self.emit_message(msg)
            raise e

        msg = str(self._wrapped_exp.name) + " : Test completed."
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
        callbacks = kwargs.pop("callbacks", [])
        callbacks.append(SlackFoldCallback(self))

        if self._notify_epochs is not None:
            callbacks.append(SlackEpochsCallback(self._notify_epochs,
                                                 self))

        kwargs["callbacks"] = callbacks

        msg = str(self._wrapped_exp.name) + " : Kfold started."
        self.emit_message(msg)

        try:
            out = self._wrapped_exp.kfold(*args, **kwargs)
        except Exception as e:
            msg = str(self._wrapped_exp.name) + " : Kfold failed. \n" + str(e)
            self.emit_message(msg)
            raise e

        msg = str(self._wrapped_exp.name) + " : Kfold completed."
        self.emit_message(msg)

        return out
