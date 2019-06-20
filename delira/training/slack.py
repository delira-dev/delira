from slackclient import SlackClient


class SlackExperiment(object):
    def __init__(self, experiment, token, channel, *args, **kwargs):
        # wrap the object
        self._client = SlackClient(token, *args, **kwargs)
        self._channel = channel
        self._wrapped_exp = experiment

        msg = "Created new experiment: " + str(self._wrapped_exp.name)
        self._client.api_call(
            "chat.postMessage",
            channel=self._channel,
            text=msg
        )

    def __getattr__(self, attr):
        # see if this object has attr
        # NOTE do not use hasattr, it goes into
        # infinite recurrsion
        if attr in self.__dict__:
            # this object has it
            return getattr(self, attr)
        # proxy to the wrapped object
        return getattr(self._wrapped_exp, attr)

    def run(self, *args, **kwargs):
        msg = str(self._wrapped_exp.name) + " : Training started."
        self._client.api_call(
            "chat.postMessage",
            channel=self._channel,
            text=msg
        )

        out = self._wrapped_exp.run(*args, **kwargs)

        msg = str(self._wrapped_exp.name) + " : Training ended."
        self._client.api_call(
            "chat.postMessage",
            channel=self._channel,
            text=msg
        )
        return out

    def resume(self, *args, **kwargs):
        msg = str(self._wrapped_exp.name) + " : Resume started."
        self._client.api_call(
            "chat.postMessage",
            channel=self._channel,
            text=msg
        )

        out = self._wrapped_exp.resume(*args, **kwargs)

        msg = str(self._wrapped_exp.name) + " : Resume ended."
        self._client.api_call(
            "chat.postMessage",
            channel=self._channel,
            text=msg
        )
        return out

    def test(self, *args, **kwargs):
        msg = str(self._wrapped_exp.name) + " : Test started."
        self._client.api_call(
            "chat.postMessage",
            channel=self._channel,
            text=msg
        )

        out = self._wrapped_exp.test(*args, **kwargs)

        msg = str(self._wrapped_exp.name) + " : Test ended."
        self._client.api_call(
            "chat.postMessage",
            channel=self._channel,
            text=msg,
        )
        return out

    def kfold(self, *args, **kwargs):
        msg = str(self._wrapped_exp.name) + " : Kfold started."
        self._client.api_call(
            "chat.postMessage",
            channel=self._channel,
            text=msg,
        )

        out = self._wrapped_exp.kfold(*args, **kwargs)

        msg = str(self._wrapped_exp.name) + " : Kfold ended."
        self._client.api_call(
            "chat.postMessage",
            channel=self._channel,
            text=msg,
        )

        return out
