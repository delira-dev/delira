from trixi.util import Config
from nested_lookup import nested_lookup
from typing import Optional


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

        if type(value) == dict or type(value) == Config:
            new_config = LookupConfig()
            new_config.update(value, deep=False)
            super().__setattr__(key, new_config)
        else:
            super().__setattr__(key, value)
