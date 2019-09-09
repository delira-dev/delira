import copy
from delira._version import get_versions
from delira.utils.time import now
from nested_lookup import nested_lookup
import warnings
from .codecs import Encoder, Decoder

import yaml
import argparse


def non_string_warning(func):
    def warning_wrapper(config, key, *args, **kwargs):
        """
        Emit warning if non string keys are used

        Parameters
        ----------
        config: :class:`Config`
            decorated function receive `self` as first argument
        key : immutable type
            key which is checked

        Returns
        -------
        callable
            original function with arguments
        """
        if not isinstance(key, str):
            warnings.warn("The key {} is not a string, but a {}. "
                          "This may lead to unwanted behavior!".format(
                              key, type(key)), RuntimeWarning)

        return func(config, key, *args, **kwargs)
    return warning_wrapper


class Config(dict):
    """
    Baseclass to create a config which hold arbitrary data
    """

    def __init__(self, dict_like=None, **kwargs):
        """

        Parameters
        ----------
        dict_like : dict, optional
            dict like object to initialize config, by default None
        kwargs:
            additional arguments added to the config

        Warnings
        --------
        If the short form of the nested settings is used integers and str
        can not be differentiated. Therefore, integers should be avoided.

        Examples
        --------
        Create simple configuration with nested keys
        >>> from delira.utils import Config
        >>> cf = Config()
        >>> # automatically generates new nested dictionaries
        >>> cf['first_level.second_level.third_level'] = 1
        >>> # form access
        >>> print(cf['first_level.second_level.third_level'])
        >>> # traditional access
        >>> print(cf['first_level']['second_level']['third_level'])
        >>> # entries can also be accessed with dot operator
        >>> print(cf.first_level.second_level.thirs_level)
        """

        super().__init__()
        self.__dict__ = self
        if dict_like is not None:
            self.update(dict_like)
        self.update(kwargs)

    @non_string_warning
    def __setattr__(self, key, value):
        """
        Set attribute in config

        Parameters
        ----------
        key : str
            attribute name
        value : any
            attribute value

        """
        super().__setattr__(key, self._to_config(value))

    @non_string_warning
    def __setitem__(self, key, value):
        """
        Set items inside dict. Supports setting of nested entries by
        seperating the individual keys with a '.'.

        Parameters
        ----------
        key : str
            key for new value
        value : any
            new value
        """
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
        """
        Get single item

        Parameters
        ----------
        key : str
            key to desired item

        Returns
        -------
        Any
            value inside dict
        """
        if not isinstance(key, str) or '.' not in key:
            try:
                return super().__getitem__(int(key))
            except (KeyError, ValueError):
                return super().__getitem__(key)
        else:
            return self._traverse_keys(key.split("."), create=False)

    @non_string_warning
    def __contains__(self, key):
        """
        Check if key is in config
        (also works for nested dicts with short form)

        Parameters
        ----------
        key : str
            key for desired value

        Returns
        -------
        bool
            true if key is in config
        """
        contain = True
        try:
            self[key]
        except KeyError:
            contain = False
        return contain

    def __str__(self):
        """
        String representation of config

        Returns
        -------
        str
            representation of config
        """
        return self.dumps()

    def _traverse_keys(self, keys, create=False):
        """
        Internal helper to traverse through nested dicts
        (iterative implementation to avoid problems with python stack)

        Parameters
        ----------
        keys : iterable of list
            iterable with keys which should be traversed
        create : bool, optional
            creates new empty configs for non existant keys, by default False

        Returns
        -------
        Any
            value defined by the traversed keys
        """
        current_level = self
        for k in keys:
            if k not in current_level:
                current_level[k] = self._create_internal_dict()
            # traverse to needed dict
            current_level = current_level._try_key(k)
        return current_level

    def _try_key(self, key):
        """
        Try access with integer key first and than with a string key

        Parameters
        ----------
        key : str or int
            key which should be tried

        Returns
        -------
        Any
            value inside config
        """
        try:
            return self[int(key)]
        except (KeyError, ValueError):
            return self[key]

    def update(self, update_dict, deepcopy=False, overwrite=False):
        """
        Update internal dicts with dict like object

        Parameters
        ----------
        update_dict : dictlike
            values which should be added to config
        deepcopy : bool, optional
            copies values from `update_dict`, by default False
        overwrite : bool, optional
            overwrite existing values inside config, by default False

        Raises
        ------
        ValueError
            if overwrite is not enabled and `update_dict` contains same values
            as config
        """
        for key, item in update_dict.items():
            # check for overwrite
            self._raise_overwrite(key, overwrite=overwrite)
            # update items individually
            self._update(key, item, deepcopy=deepcopy, overwrite=overwrite)

    def _update(self, key, item, deepcopy=False, overwrite=False):
        """
        Helper function for update

        Parameters
        ----------
        key : str
            key where new item should be assigned
        item : Any
            item which should be assigned
        deepcopy : bool, optional
            copies values from `update_dict`, by default False
        overwrite : bool, optional
            overwrite existing values inside config, by default False
        """
        if isinstance(item, dict):
            # update nested dicts
            if key not in self:
                self[key] = self._create_internal_dict({})
            self[key].update(item, deepcopy=deepcopy, overwrite=overwrite)
        else:
            # set item
            self._set_internal_item(key, item, deepcopy=deepcopy)

    def _raise_overwrite(self, key, overwrite):
        """
        Checks if a ValueError should be raised

        Parameters
        ----------
        key : str
            key which needs to be checked
        overwrite : bool
            if overwrite ir enables no error is raised even if key already
            exists

        Raises
        ------
        ValueError
            if overwrite is not enabled and key already exists
        """
        if key in self and not overwrite:
            raise ValueError("{} already in config. Can "
                             "not overwrite value.".format(key))

    def _set_internal_item(self, key, item, deepcopy=False):
        """
        Set internal item

        Parameters
        ----------
        key : str
            key where new item should be assigned
        item : Any
            item which should be assigned
        deepcopy : bool, optional
            if enabled the item is copied to the config, by default False
        """
        config_item = self._to_config(item)
        if deepcopy:
            self[key] = copy.deepcopy(config_item)
        else:
            self[key] = config_item

    @staticmethod
    def _create_internal_dict(*args, **kwargs):
        """
        Defines how internal dicts should be created. Can be used to easily
        overwrite subclasses

        Returns
        -------
        :class:`Config`
            new config
        """
        return Config(*args, **kwargs)

    @classmethod
    def _to_config(cls, item):
        """
        Convert items to config if they are a dict like object
        but not already a config

        Parameters
        ----------
        item : Any
            item which is converted

        Returns
        -------
        Any
            return a config is item is dict like, otherwise the item is
            returned
        """
        if isinstance(item, dict) and not isinstance(item, cls):
            # convert dict to config for additional functionality
            return cls._create_internal_dict(item)
        else:
            return item

    def dump(self, path, formatter=yaml.dump, encoder_cls=Encoder, **kwargs):
        """
        Save config to a file and add time stamp to config

        Parameters
        ----------
        path : str
            path where config is saved
        formatter : callable, optional
            defines the format how the config is saved, by default yaml.dump
        encoder_cls : :class:`Encoder`, optional
            trasforms config to a format which can be formatted by the
            `formatter`, by default Encoder
        kwargs:
            additional keyword arguments passed to `formatter`
        """
        self._timestamp = now()
        encoded_self = encoder_cls().encode(self)
        with open(path, "w") as f:
            formatter(encoded_self, f, **kwargs)

    def dumps(self, formatter=yaml.dump, encoder_cls=Encoder, **kwargs):
        """
        Create a loadable string representation from the config and
        add time stamp to config

        Parameters
        ----------
        formatter : callable, optional
            defines the format how the config is saved, by default yaml.dump
        encoder_cls : :class:`Encoder`, optional
            trasforms config to a format which can be formatted by the
            `formatter`, by default Encoder
        kwargs:
            additional keyword arguments passed to `formatter`
        """
        self._timestamp = now()
        encoded_self = encoder_cls().encode(self)
        return formatter(encoded_self, **kwargs)

    @classmethod
    def load(cls, path, formatter=yaml.load, decoder_cls=Decoder, **kwargs):
        """
        Load config from a file

        Parameters
        ----------
        path : str
            path to file
        formatter : callable, optional
            defines the format how the config is saved, by default yaml.dump
        encoder_cls : :class:`Encoder`, optional
            trasforms config to a format which can be formatted by the
            `formatter`, by default Encoder
        kwargs:
            additional keyword arguments passed to `formatter`
        """
        with open(path, "r") as f:
            decoded_format = formatter(f, **kwargs)
        decoded_format = decoder_cls().decode(decoded_format)
        if not isinstance(decoded_format, cls):
            decoded_format = cls(decoded_format)
        return decoded_format

    @classmethod
    def loads(cls, data, formatter=yaml.load, decoder_cls=Decoder, **kwargs):
        """
        Load config from a string

        Parameters
        ----------
        formatter : callable, optional
            defines the format how the config is saved, by default yaml.dump
        encoder_cls : :class:`Encoder`, optional
            trasforms config to a format which can be formatted by the
            `formatter`, by default Encoder
        kwargs:
            additional keyword arguments passed to `formatter`
        """
        decoded_format = formatter(data, **kwargs)
        decoded_format = decoder_cls().decode(decoded_format)
        if not isinstance(decoded_format, cls):
            decoded_format = cls(decoded_format)
        return decoded_format

    @classmethod
    def create_from_dict(cls, value, deepcopy=False):
        """
        Create config from dict like object

        Parameters
        ----------
        value : dict like
            dict like object used to create new config
        deepcopy : bool, optional
            if enabled, copies values from origin, by default False

        Returns
        -------
        :class:`Config`
            new config
        """
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
        """
        Create config from argument parser

        Parameters
        ----------
        value : argument parser or namespace
            if value is an argument parser, the arguments are first parsed
            and than a new config with the values is created
            if value is a Namespace the new config is created immediatly
        deepcopy : bool, optional
            if enabled, copies values from origin, by default False

        Returns
        -------
        :class:`Config`
            new config

        Raises
        ------
        TypeError
            if value is not an instance of :class:`ArgumentParser`
            or :class:`Namespace`
        """
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
        """
        Defines how internal dicts should be created. Can be used to easily
        overwrite subclasses

        Returns
        -------
        :class:`LookupConfig`
            new config
        """
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
    """
    Configure experiment for delira. Contains variables for model and training
    which can be either fixed or variables (for hyperparameter search)
    """

    def __init__(self, dict_like=None, fixed_model=None, fixed_training=None,
                 variable_model=None, variable_training=None, **kwargs):
        """

        Parameters
        ----------
        dict_like : dict, optional
            dict like object containing values for config, by default None.
        fixed_model : dict, optional
            fixed parameters for model, by default None.
        fixed_training : dict, optional
            fixed parameters for training, by default None.
        variable_model : dict, optional
            variable parameters for model, by default None.
        variable_training : dict, optional
            variable parameters for training, by default None.
        kwargs:
            additional arguments added to the config
        """
        super().__init__(dict_like=dict_like, **kwargs)
        self._update("fixed_model", self.generate_dict(fixed_model),
                     overwrite=True)
        self._update("fixed_training", self.generate_dict(fixed_training),
                     overwrite=True)
        self._update("variable_model", self.generate_dict(variable_model),
                     overwrite=True)
        self._update(
            "variable_training",
            self.generate_dict(variable_training),
            overwrite=True)
        self._version = get_versions()["version"]

    @staticmethod
    def generate_dict(value):
        """
        If value is none an emty dict will be created

        Parameters
        ----------
        value : Any
            checked value

        Returns
        -------
        Any
            dict if value is none otherwise value is returned
        """
        if value is None:
            return {}
        else:
            return value

    @property
    def params(self):
        """
        Returns a :class:`LookupConfig` with all model and training parameters

        Returns
        -------
        :class:`LookupConfig`
            config with model and training parameters
        """
        return LookupConfig(fixed_model=self.fixed_model,
                            fixed_training=self.fixed_training,
                            variable_model=self.variable_model,
                            variable_training=self.variable_training)

    @property
    def variable_params(self):
        """
        Returns a :class:`LookupConfig` with all variable parameters

        Returns
        -------
        :class:`LookupConfig`
            config with variable parameters
        """
        return LookupConfig(model=self.variable_model,
                            training=self.variable_training)

    @variable_params.setter
    def variable_params(self, new_params: dict):
        """
        Update variable parameters from dict like object
        """
        # create empty dict
        if "model" not in new_params:
            new_params["model"] = {}

        # create empty dict
        if "training" not in new_params:
            new_params["training"] = {}

        self.variable_model.update(new_params["model"], overwrite=True)
        self.variable_training.update(new_params["training"], overwrite=True)

    @property
    def fixed_params(self):
        """
        Returns a :class:`LookupConfig` with all fixed parameters

        Returns
        -------
        :class:`LookupConfig`
            config with fixed parameters
        """
        return LookupConfig(model=self.fixed_model,
                            training=self.fixed_training)

    @fixed_params.setter
    def fixed_params(self, new_params: dict):
        """
        Update fixed parameters from dict like object
        """
        # create empty dict
        if "model" not in new_params:
            new_params["model"] = {}

        # create empty dict
        if "training" not in new_params:
            new_params["training"] = {}

        self.fixed_model.update(new_params["model"], overwrite=True)
        self.fixed_training.update(new_params["training"], overwrite=True)

    @property
    def model_params(self):
        """
        Returns a :class:`LookupConfig` with all model parameters

        Returns
        -------
        :class:`LookupConfig`
            config with model parameters
        """
        return LookupConfig(variable=self.variable_model,
                            fixed=self.fixed_model)

    @model_params.setter
    def model_params(self, new_params: dict):
        """
        Update model parameters from dict like object
        """
        # create empty dict
        if "fixed" not in new_params:
            new_params["fixed"] = {}

        # create empty dict
        if "variable" not in new_params:
            new_params["variable"] = {}

        self.fixed_model.update(new_params["fixed"], overwrite=True)
        self.variable_model.update(new_params["variable"], overwrite=True)

    @property
    def training_params(self):
        """
        Returns a :class:`LookupConfig` with all training parameters

        Returns
        -------
        :class:`LookupConfig`
            config with training parameters
        """
        return LookupConfig(variable=self.variable_training,
                            fixed=self.fixed_training)

    @training_params.setter
    def training_params(self, new_params: dict):
        """
        Update training parameters from dict like object
        """
        # create empty dict
        if "fixed" not in new_params:
            new_params["fixed"] = {}

        # create empty dict
        if "variable" not in new_params:
            new_params["variable"] = {}

        self.fixed_training.update(new_params["fixed"], overwrite=True)
        self.variable_training.update(new_params["variable"], overwrite=True)

    def log_as_string(self, full_config=False, **kwargs):
        """
        Log current config as a string

        Parameters
        ----------
        full_config : bool, optional
            if enabled the complete Config is logged, by default False.
            Otherwise only model and training parameters will be logged.
        kwargs:
            keyword arguments passed to `self.dumps` method to create string
            representation

        Returns
        -------
        str
            string representation used for logging
        """
        from delira.logging import log

        if full_config:
            str_repr = self.dumps(**kwargs)
        else:
            str_repr = self.params.dumps(**kwargs)
        log({'text': {"text_string": str_repr, "tag": "DeliraConfig"}})
        return str_repr
