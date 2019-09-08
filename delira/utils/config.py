import copy
from delira._version import get_versions
from nested_lookup import nested_lookup
import warnings
from .codecs import Encoder, Decoder

import yaml
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
        super().__setattr__(key, self._to_config(value))

    @non_string_warning
    def __setitem__(self, key, value):
        if not isinstance(key, str) or '.' not in key:
            super().__setitem__(key, value)
        else:
            current_level = self
            keys = key.split(".")
            final_key = keys.pop(-1)
            final_dict = self._traverse_keys(keys, create=True)
            final_dict._set_internal_item(final_key, value)

    @non_string_warning
    def __getitem__(self, key):
        if not isinstance(key, str) or '.' not in key:
            try:
                return super().__getitem__(int(key))
            except (KeyError, ValueError):
                return super().__getitem__(key)
        else:
            return self._traverse_keys(key.split("."), create=False)

    @non_string_warning
    def __contains__(self, key):
        contain = True
        try:
            self[key]
        except KeyError:
            contain = False
        return contain

    def __str__(self):
        return self.dumps()

    def _traverse_keys(self, keys, create=False):
        current_level = self
        for k in keys:
            if k not in current_level:
                current_level[k] = self._create_internal_dict()
            # traverse to needed dict
            current_level = current_level._try_key(k)
        return current_level

    def _try_key(self, key):
        try:
            return self[int(key)]
        except (KeyError, ValueError):
            return self[key]

    def update(self, update_dict, deepcopy=False, overwrite=False):
        for key, item in update_dict.items():
            # check for overwrite
            self._raise_overwrite(key, overwrite=overwrite)
            # update items individually
            self._update(key, item, deepcopy=deepcopy, overwrite=overwrite)

    def _update(self, key, item, deepcopy=False, overwrite=False):
        if isinstance(item, dict):
            # update nested dicts
            if key not in self:
                self[key] = self._create_internal_dict({})
            self[key].update(item, deepcopy=deepcopy, overwrite=overwrite)
        else:
            # set item
            self._set_internal_item(key, item, deepcopy=deepcopy)

    def _raise_overwrite(self, key, overwrite):
        if key in self and not overwrite:
            raise ValueError("{} already in config. Can "
                             "not overwrite value.".format(key))

    def _set_internal_item(self, key, item, deepcopy=False):
        config_item = self._to_config(item)
        if deepcopy:
            self[key] = copy.deepcopy(config_item)
        else:
            self[key] = config_item

    @staticmethod
    def _create_internal_dict(*args, **kwargs):
        return Config(*args, **kwargs)

    @classmethod
    def _to_config(cls, item):
        if isinstance(item, dict) and not isinstance(item, cls):
            # convert dict to config for additional functionality
            return cls._create_internal_dict(item)
        else:
            return item

    def dump(self, path, formatter=yaml.dump, encoder_cls=Encoder, **kwargs):
        encoded_self = encoder_cls().encode(self)
        with open(path, "w") as f:
            formatter(encoded_self, f, **kwargs)

    def dumps(self, formatter=yaml.dump, encoder_cls=Encoder, **kwargs):
        encoded_self = encoder_cls().encode(self)
        return formatter(encoded_self, **kwargs)

    @classmethod
    def load(cls, path, formatter=yaml.load, decoder_cls=Decoder, **kwargs):
        with open(path, "r") as f:
            decoded_format = formatter(f, **kwargs)
        return decoder_cls().decode(decoded_format)

    @classmethod
    def loads(cls, data, formatter=yaml.load, decoder_cls=Decoder, **kwargs):
        decoded_format = formatter(data, **kwargs)
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

    @staticmethod
    def _create_internal_dict(*args, **kwargs):
        return LookupConfig(*args, **kwargs)

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
        self._version = get_versions()["version"]

    @property
    def params(self):
        return LookupConfig(fixed_model=self.fixed_model,
                            fixed_training=self.fixed_training,
                            variable_model=self.variable_model,
                            variable_training=self.variable_training)

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

    def log_as_string(self, full_config=False, **kwargs):
        from delira.logging import log

        if full_config:
            str_repr = self.dumps(**kwargs)
        else:
            str_repr = self.params.dumps(**kwargs)
        log({'text': {"text_string": str_repr, "tag": "DeliraConfig"}})
        return str_repr
