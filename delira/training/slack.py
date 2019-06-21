import logging
from slackclient import SlackClient

from delira.training import BaseExperiment
from delira.training.callbacks import AbstractCallback


class SlackEpochsCallback(AbstractCallback):
    def __init__(self, n_epochs, slack_exp):
        super().__init__()
        self._n_epochs = n_epochs
        self._slack_exp = slack_exp

    def at_epoch_end(self, trainer, **kwargs):
        curr_epoch = kwargs.pop("curr_epoch")
        trained_epochs = curr_epoch - trainer.start_epoch
        if trained_epochs % self._n_epochs == 0:
            msg = "Epoch " + str(curr_epoch) + " trained."
            self._slack_exp.emit_message(msg)
        return {}


class SlackFoldCallback(AbstractCallback):
    def __init__(self, slack_exp):
        super().__init__()
        self._slack_exp = slack_exp

    def at_training_end(self, trainer, **kwargs):
        msg = "Fold " + str(trainer.fold) + " completed."
        self._slack_exp.emit_message(msg)
        return {}


class SlackExperiment(object):
    def __init__(self, experiment: BaseExperiment, token: str,
                 channel: str, *args, notify_epochs=None, **kwargs):
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
        # see if this object has attr
        # NOTE do not use hasattr, it goes into
        # infinite recurrsion
        if attr in self.__dict__:
            # this object has it
            return getattr(self, attr)
        # proxy to the wrapped object
        return getattr(self._wrapped_exp, attr)

    def emit_message(self, msg):
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
