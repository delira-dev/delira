import copy
from delira.utils.time import now
from delira._version import get_versions
from nested_lookup import nested_lookup
import warnings
from .codecs import Encoder, Decoder

import json
import argparse


def non_string_warning(func):
    def warning_wrapper(key, *args, **kwargs):
        # if not isinstance(key, str):
        #     warnings.warn("The key {} is not a string, but a {}. "
        #                   "This may lead to unwanted behavior upon encoding "
        #                   "and decoding!".format(key, type(key)),
        #                   RuntimeWarning)

        return func(key, *args, **kwargs)

    return warning_wrapper


class Config(dict):
    def __init__(self, dict_like=None, **kwargs):
        super().__init__()
        self.__dict__ = self
        if dict_like is not None:
            self.update(dict_like)
        self.update(kwargs)

    @non_string_warning
    def __setattr__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, type(self)):
            # convert dict to config for additional functionality
            value = Config(value)
        super().__setattr__(key, value)

    @non_string_warning
    def __setitem__(self, key, value):
        if not isinstance(key, str) or '.' not in key:
            super().__setitem__(key, value)
        else:
            current_level = self
            key_split = key.split(".")
            final_key = key_split.pop(-1)
            for k in key_split:
                # traverse to needed dict
                if k not in current_level:
                    new_config = Config()
                    current_level[k] = new_config
                current_level = current_level[k]
            current_level[final_key] = value

    @non_string_warning
    def __getitem__(self, key):
        if not isinstance(key, str) or '.' not in key:
            try:
                return super().__getitem__(int(key))
            except (KeyError, ValueError):
                return super().__getitem__(key)
        else:
            current_level = self
            key_split = key.split(".")
            final_key = key_split.pop(-1)
            for k in key_split:
                # traverse to needed dict
                try:
                    current_level = current_level[int(k)]
                except (KeyError, ValueError):
                    current_level = current_level[k]
            try:
                return current_level[int(final_key)]
            except (KeyError, ValueError):
                return current_level[final_key]

    @non_string_warning
    def __contains__(self, key):
        if not isinstance(key, str) or '.' not in key:
            return super().__contains__(key)
        else:
            current_level = self
            key_split = key.split(".")
            final_key = key_split.pop(-1)
            for k in key_split:
                # traverse to needed dict
                if isinstance(current_level, dict) and (k in current_level):
                    current_level = current_level[k]
                else:
                    return False
            return (final_key in current_level)

    def update(self, update_dict, deepcopy=False, overwrite=False):
        for key, item in update_dict.items():
            if key in self:
                if isinstance(item, dict):
                    self[key].update(update_dict[key],
                                     deepcopy=deepcopy,
                                     overwrite=overwrite)
                else:
                    if overwrite:
                        if deepcopy:
                            self[key] = copy.deepcopy(item)
                        else:
                            self[key] = item
                    else:
                        raise ValueError("{} already in config. Can "
                                         "not overwrite value.".format(key))
            else:
                if isinstance(item, dict) and not isinstance(item, type(self)):
                    self[key] = Config(item)
                else:
                    if deepcopy:
                        self[key] = copy.deepcopy(item)
                    else:
                        self[key] = item

    def dump(self, path, format_fn=json.dump, encoder_cls=Encoder, **kwargs):
        encoded_self = encoder_cls().encode(self)
        with open(path, "w") as f:
            format_fn(encoded_self, f, **kwargs)

    def dumps(self, format_fn=json.dumps, encoder_cls=Encoder, **kwargs):
        encoded_self = encoder_cls().encode(self)
        return format_fn(encoded_self, **kwargs)

    @classmethod
    def load(cls, path, format_fn=json.load, decoder_cls=Decoder, **kwargs):
        with open(path, "r") as f:
            decoded_format = format_fn(f, **kwargs)
        return decoder_cls().decode(decoded_format)

    @classmethod
    def loads(cls, data, format_fn=json.loads, decoder_cls=Decoder, **kwargs):
        decoded_format = format_fn(data, **kwargs)
        return decoder_cls().decode(decoded_format)

    @classmethod
    def create_from_dict(cls, value, deepcopy=False):
        assert isinstance(value, dict)
        new_config = cls()
        for key, item in value.items():
            if isinstance(item, dict):
                item = cls(item)
            if deepcopy:
                new_config[key] = copy.deepcopy(item)
            else:
                new_config[key] = item
        return new_config

    @classmethod
    def create_from_argparse(cls, value, deepcopy=False, **kwargs):
        if isinstance(value, argparse.ArgumentParser):
            args_parsed = value.parse_args(**kwargs)
            return cls.create_from_dict(vars(args_parsed), deepcopy=deepcopy)
        elif isinstance(value, argparse.Namespace):
            return cls.create_from_dict(vars(value), deepcopy=deepcopy)
        else:
            raise TypeError("Type of args not supported.")


class LookupConfig(Config):
    """
    Helper class to have nested lookups in all subdicts of Config

    """

    def nested_get(self, key, *args, **kwargs):
        """
        Returns all occurances of ``key`` in ``self`` and subdicts

        Parameters
        ----------
        key : str
            the key to search for
        *args :
            positional arguments to provide default value
        **kwargs :
            keyword arguments to provide default value

        Raises
        ------
        KeyError
            Multiple Values are found for key
            (unclear which value should be returned)
            OR
            No Value was found for key and no default value was given

        Returns
        -------
        Any
            value corresponding to key (or default if value was not found)

        """

        if "." in key:
            return self[key]
        results = nested_lookup(key, self)
        if len(results) > 1:
            raise KeyError("Multiple Values found for key %s" % key)
        elif len(results) == 0:
            if "default" in kwargs:
                return kwargs["default"]
            elif args:
                return args[0]
            else:
                raise KeyError("No Value found for key %s" % key)
        else:
            return results[0]


class DeliraConfig(LookupConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_model = LookupConfig()
        self.fixed_training = LookupConfig()
        self.variable_model = LookupConfig()
        self.variable_training = LookupConfig()
        self._timestamp = now()
        self._version = get_versions()["version"]

    @property
    def variable_params(self):
        return LookupConfig(model=self.variable_model,
                            training=self.variable_training)

    @variable_params.setter
    def variable_params(self, new_params: dict):

        # create empty dict
        if "model" not in new_params:
            new_params["model"] = {}

        # create empty dict
        if "training" not in new_params:
            new_params["training"] = {}

        self.variable_model.update(new_params["model"])
        self.variable_training.update(new_params["training"])

    @property
    def fixed_params(self):
        return LookupConfig(model=self.fixed_model,
                            training=self.fixed_training)

    @fixed_params.setter
    def fixed_params(self, new_params: dict):
        # create empty dict
        if "model" not in new_params:
            new_params["model"] = {}

        # create empty dict
        if "training" not in new_params:
            new_params["training"] = {}

        self.fixed_model.update(new_params["model"])
        self.fixed_training.update(new_params["training"])

    @property
    def model_params(self):
        return LookupConfig(variable=self.variable_model,
                            fixed=self.fixed_model)

    @model_params.setter
    def model_params(self, new_params: dict):
        # create empty dict
        if "fixed" not in new_params:
            new_params["fixed"] = {}

        # create empty dict
        if "variable" not in new_params:
            new_params["variable"] = {}

        self.fixed_model.update(new_params["fixed"])
        self.variable_model.update(new_params["variable"])

    @property
    def training_params(self):
        return LookupConfig(variable=self.variable_training,
                            fixed=self.fixed_training)

    @training_params.setter
    def training_params(self, new_params: dict):

        # create empty dict
        if "fixed" not in new_params:
            new_params["fixed"] = {}

        # create empty dict
        if "variable" not in new_params:
            new_params["variable"] = {}

        self.fixed_training.update(new_params["fixed"])
        self.variable_training.update(new_params["variable"])

    # TODO: logging as string
    def log_as_string(self):
        raise NotImplementedError

    # TODO: logging as hyperparameters
    def log_as_hyperparameter(self):
        raise NotImplementedError

    # TODO: overwrite class in LookupConfig for creation
