import copy

from nested_lookup import nested_lookup
from trixi.util import Config

# TODO: check deepcopy


class BaseConfig(dict):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.__dict__ = self

    def __setattr__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, BaseConfig):
            # convert dict to config for additional funtionality
            value = BaseConfig.create_from_dict(value)
        super().__setattr__(key, value)

    def __setitem__(self, key, value):
        if not '.' in key:
            super().__setitem__(key, value)
        else:
            current_level = self
            key_split = key.split(".")
            final_key = key_split.pop(-1)
            for k in key_split:
                # traverse to needed dict
                if k not in current_level:
                    new_config = BaseConfig()
                    current_level[k] = new_config
                current_level = current_level[k]
            current_level[final_key] = value

    def __getitem__(self, key):
        if not '.' in key:
            return super().__getitem__(key)
        else:
            current_level = self
            key_split = key.split(".")
            final_key = key_split.pop(-1)
            for k in key_split:
                # traverse to needed dict
                current_level = current_level[k]
            return current_level[final_key]

    def __contains__(self, key):
        if not '.' in key:
            return super().__contains__(key)
        else:
            current_level = self
            key_split = key.split(".")
            final_key = key_split.pop(-1)
            for k in key_split:
                # traverse to needed dict
                if isinstance(current_level, dict) or k not in current_level:
                    return False
                else:
                    current_level = current_level[k]
            return (final_key in current_level)

    @staticmethod
    def create_from_dict(value, val_copy=False):
        # TODO: maybe remove recursion
        assert isinstance(value, dict)
        new_config = BaseConfig()
        for key, item in value.items():
            if isinstance(item, dict):
                item = BaseConfig.create_from_dict(item, val_copy=val_copy)
            if val_copy:
                new_config[key] = copy.deepcopy(item)
            else:
                new_config[key] = item
        return new_config


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

    def __setattr__(self, key, value):
        """
        Modified to automatically convert `dict` to LookupConfig.

        Parameters
        ----------
        key : str
            the key which should be set
        value : Any
            the corresponding value to set the key to.

        """

        if isinstance(value, dict) and not isinstance(value, LookupConfig):
            value = LookupConfig(config=value)

        return super().__setattr__(key, value)
