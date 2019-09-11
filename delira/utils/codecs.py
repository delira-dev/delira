import importlib
import types
import collections
import inspect
import numpy as np
import logging
import typing
from functools import partial
import typing


class Encoder:
    """
    Encode arbitrary objects. The encoded object consists of dicts,
    lists, ints, floats and strings.
    """

    def __call__(self, obj) -> typing.Any:
        """
        Encode arbitrary objects as dicts, str, int, float, list

        Parameters
        ----------
        obj : Any
            object to be encoded

        Returns
        -------
        Any
            encoded object
        """
        return self.encode(obj)

    def encode(self, obj) -> typing.Any:
        """
        Encode arbitrary objects as dicts, str, int, float, list

        Parameters
        ----------
        obj : Any
            object to be encoded

        Returns
        -------
        Any
            encoded object
        """
        # use type() to check for dict and list because type() does not
        # consider subtypes which is the desired behaviour in this case
        if isinstance(obj, (str, int, float)):
            # end recursion
            return obj
        elif obj is None:
            return obj
        elif type(obj) == dict:
            # end recursion
            return self._encode_dict(obj)
        elif type(obj) == list:
            # end recursion
            return self._encode_list(obj)
        elif isinstance(obj, np.ndarray):
            return self._encode_array(obj)
        elif isinstance(obj, collections.Mapping):
            return self._encode_mapping(obj)
        elif isinstance(obj, collections.Iterable):
            return self._encode_iterable(obj)
        elif isinstance(obj, types.ModuleType):
            return self._encode_module(obj)
        elif inspect.isclass(obj) or isinstance(obj, type):
            # use both ways to determine functions here
            # (the second uglier one serves as fallback here in case inspect
            # does not cover all cases)
            return self._encode_type(obj)
        elif isinstance(obj, (types.BuiltinFunctionType, types.FunctionType)):
            return self._encode_function(obj)
        else:
            return self._encode_class(obj)

    def _encode_list(self, obj) -> list:
        """
        Encode list

        Parameters
        ----------
        obj : list
            list to be encoded

        Returns
        -------
        list
            list with encoded internal items
        """
        return [self.encode(i) for i in obj]

    def _encode_dict(self, obj) -> dict:
        """
        Encode dict

        Parameters
        ----------
        obj : dict
            dict to be encoded

        Returns
        -------
        dict
            dict with encoded internal items
        """
        return {self.encode(_key):
                self.encode(_item) for _key, _item in obj.items()}

    def _encode_array(self, obj) -> dict:
        """
        Encode array

        Parameters
        ----------
        obj : :class:`np.ndarray`
            object to be encoded

        Returns
        -------
        dict
            array encoded as a list inside a dict
        """
        # # if numpy array: add explicit array specifier
        # use tolist instead of tostring here (even though this requires
        # additional encoding steps and increases memory usage), since tolist
        # retains the shape and tostring doesn't
        return {"__array__": self.encode(obj.tolist())}

    def _encode_mapping(self, obj) -> dict:
        """
        Encode mapping

        Parameters
        ----------
        obj : collections.Mapping
            object to be encoded

        Returns
        -------
        dict
            mapping encoded as a dict with original data and type
        """
        # encode via encoding the type and the mapping converted to dict
        # separately and add a conversion specifier
        convert_repr = {
            "type": self.encode(type(obj)),
            "repr": self.encode(dict(obj)),
        }
        return {"__convert__": convert_repr}

    def _encode_iterable(self, obj) -> dict:
        """
        Encode iterable

        Parameters
        ----------
        obj : collections.Iterable
            object to be encoded

        Returns
        -------
        dict
            iterable encoded as a dict with original data and type
        """
        # encode via converting the type and the mapping converted to list
        # separately and add conversion specifier
        convert_repr = {
            "type": self.encode(type(obj)),
            "repr": self.encode(list(obj)),
        }
        return {"__convert__": convert_repr}

    def _encode_module(self, obj) -> dict:
        """
        Encode module

        Parameters
        ----------
        obj : types.ModuleType
            module to be encoded

        Returns
        -------
        dict
            module encoded as a dict
        """
        # encode via name and module specifier
        return {"__module__": obj.__module__}

    def _encode_type(self, obj) -> dict:
        """
        Encode class or type

        Parameters
        ----------
        obj :
            class/type to be encoded

        Returns
        -------
        dict
            class/type encoded as a dict
        """
        type_repr = {
            "module": self.encode(obj.__module__),
            "name": self.encode(obj.__name__),
        }
        return {"__type__": type_repr}

    def _encode_function(self, obj) -> dict:
        """
        Encode function

        Parameters
        ----------
        obj :
            function to be encoded

        Returns
        -------
        dict
            function encoded as a dict
        """
        function_repr = {
            "module": self.encode(obj.__module__),
            "name": self.encode(obj.__name__),
        }
        return {"__function__": function_repr}

    def _encode_class(self, obj) -> dict:
        """
        Encode arbitrary object

        Parameters
        ----------
        obj :
             arbitrary object to be encoded

        Returns
        -------
        dict
             arbitrary object encoded as a dict
        """
        try:
            class_repr = {
                "type": self.encode(type(obj)),
                "dict": self.encode(obj.__dict__)
            }
            return {"__class__": class_repr}
        except Exception as e:
            logging.error(e)


class Decoder:
    """
    Deocode arbitrary objects which were encoded by :class:`Encoder`.
    """

    def __init__(self):
        super().__init__()
        self._decode_mapping = {
            "__array__": self._decode_array,
            "__convert__": self._decode_convert,
            "__module__": self._decode_module,
            "__type__": self._decode_type,
            "__function__": self._decode_function,
            "__class__": self._decode_class,
            "__classargs__": self._decode_classargs,
            "__functionargs__": self._decode_functionargs
        }

    def __call__(self, obj) -> typing.Any:
        """
        Decode object

        Parameters
        ----------
        obj : Any
            object to be decoded

        Returns
        -------
        Any
            decoded object
        """
        return self.decode(obj)

    def decode(self, obj) -> typing.Any:
        """
        Decode object

        Parameters
        ----------
        obj : Any
            object to be decoded

        Returns
        -------
        Any
            decoded object
        """
        if isinstance(obj, (str, int, float)):
            return obj
        elif isinstance(obj, dict):
            return self._decode_dict(obj)
        elif isinstance(obj, list):
            return self._decode_list(obj)
        else:
            return obj

    def _decode_dict(self, obj) -> dict:
        """
        Decode dict with respect to unique identifier keys.

        Parameters
        ----------
        obj : dict
            dict to be decoded

        Returns
        -------
        dict
            decoded dict
        """
        for key in obj.keys():
            if key in self._decode_mapping:
                return self._decode_mapping[key](obj[key])
            else:
                obj[key] = self.decode(obj[key])
        return obj

    def _decode_list(self, obj) -> list:
        """
        Decode list

        Parameters
        ----------
        obj : list
            list to be decoded

        Returns
        -------
        Any
            decoded list
        """
        return [self.decode(_i) for _i in obj]

    def _decode_array(self, obj) -> np.ndarray:
        """
        Decode np.ndarray

        Parameters
        ----------
        obj : :class:`np.ndarray`
            array to be decoded

        Returns
        -------
        :class:`np.ndarray`
            decoded array
        """
        return np.array(self.decode(obj))

    def _decode_convert(self, obj: dict) -> typing.Union[
            typing.Iterable, typing.Mapping]:
        """
        Decode mappings and iterables

        Parameters
        ----------
        obj : dict
            dict to be decoded

        Returns
        -------
        typing.Union[typing.Iterable, typing.Mapping]
            decoded object
        """
        # decode items in dict representation
        convert_repr = self.decode(obj)
        # create new object
        return convert_repr["type"](convert_repr["repr"])

    def _decode_module(self, obj: dict) -> types.ModuleType:
        """
        Decode module

        Parameters
        ----------
        obj : dict
            dict to be decoded

        Returns
        -------
        ModuleType
            decoded module
        """
        return importlib.import_module(self.decode(obj))

    def _decode_type(self, obj) -> typing.Any:
        """
        Decode type

        Parameters
        ----------
        obj : dict
            dict to be decoded

        Returns
        -------
        Any
            decoded type
        """
        # decode items in dict representation
        type_repr = self.decode(obj)
        return getattr(importlib.import_module(type_repr["module"]),
                       type_repr["name"])

    def _decode_function(self, obj: dict) -> typing.Union[
            types.FunctionType, types.BuiltinFunctionType]:
        """
        Decode function

        Parameters
        ----------
        obj : dict
            dict to be decoded

        Returns
        -------
        typing.Union[types.FunctionType, types.BuiltinFunctionType]
            decoded function
        """
        # decode items in dict representation
        function_repr = self.decode(obj)
        return getattr(importlib.import_module(function_repr["module"]),
                       function_repr["name"])

    def _decode_class(self, obj: dict) -> typing.Any:
        """
        Decode arbitrary object

        Parameters
        ----------
        obj : dict
            dict to be decoded

        Returns
        -------
        Any
            decoded object
        """
        class_repr = self.decode(obj)
        cls_type = class_repr["type"]
        cls_dict = class_repr["dict"]

        # need to create a temporary type here (which is basically a raw
        # object, since using object directly raises
        # "TypeError: __class__ assignment only supported for heap types
        # or ModuleType subclasses"
        # After a bit of research this kind of class re-creation only
        # seems to be possible, if the intermediate class was created in
        # python (which is not True for the object type since this is part
        # of Python's C Core)
        tmp_cls = type("__tmp", (), {})
        # create instance of temporary class
        tmp_instance = tmp_cls()
        # change class type
        tmp_instance.__class__ = self.decode(cls_type)
        # update attributes of class
        tmp_instance.__dict__.update(self.decode(cls_dict))
        return tmp_instance

    def _decode_classargs(self, obj: dict) -> typing.Any:
        """
        Create an object from specified class and arguments

        Parameters
        ----------
        obj : dict
            dictionary which representes the object. Must include `module` and
            `name`. Can optionally include `args` and `kwargs`.

        Returns
        -------
        Any
            decoded object

        Raises
        ------
        TypeError
            arguments and name must be encoded as a dict
        """
        classargs = self.decode(obj)

        if not isinstance(classargs, dict):
            raise TypeError("Arguments for classargs must be defined as dict.")

        obj_cls = getattr(importlib.import_module(classargs["module"]),
                          classargs["name"])
        args = classargs.get("args", [])
        kwargs = classargs.get("kwargs", {})
        return obj_cls(*args, **kwargs)

    def _decode_functionargs(self, obj: dict) -> typing.Any:
        """
        Create an function from specified function and arguments

        Parameters
        ----------
        obj : dict
            dictionary which representes the function. Must include `module`
            and `name`. Can optionally include `args` and `kwargs` which are
            passed via `functool.partial`.

        Returns
        -------
        Any
            decoded function

        Raises
        ------
        TypeError
            arguments and name must be encoded as a dict
        """
        functionargs = self.decode(obj)

        if not isinstance(functionargs, dict):
            raise TypeError("Arguments for classargs must be defined as dict.")

        fn = getattr(importlib.import_module(functionargs["module"]),
                     functionargs["name"])
        args = functionargs.get("args", [])
        kwargs = functionargs.get("kwargs", {})
        return partial(fn, args, kwargs)
