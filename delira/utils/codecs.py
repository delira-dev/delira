import importlib
import types
import collections
import inspect
import numpy as np
import logging
import typing


class Encoder:
    def __call__(self, obj) -> str:
        return self.encode(obj)

    def encode(self, obj) -> str:
        # use type() to check for dict and list because type() does not
        # consider subtypes which is the desired behaviour in this case
        if isinstance(obj, (str, int, float)):
            # end recursion
            return obj
        elif obj is None:
            return obj
        elif isinstance(obj, dict):
            # end recursion
            return self._encode_dict(obj)
        elif isinstance(obj, list):
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

    def _encode_list(self, obj) -> str:
        # if list: add explicit list specifier
        return [self.encode(i) for i in obj]

    def _encode_dict(self, obj) -> str:
        # if dict: add specific dict specifier
        return {self.encode(_key):
                self.encode(_item) for _key, _item in obj.items()}

    def _encode_array(self, obj) -> str:
        # # if numpy array: add explicit array specifier
        # use tolist instead of tostring here (even though this requires
        # additional encoding steps and increases memory usage), since tolist
        # retains the shape and tostring doesn't
        return {"__array__": self.encode(obj.tolist())}

    def _encode_mapping(self, obj) -> str:
        # encode via encoding the type and the mapping converted to dict
        # separately and add a conversion specifier
        convert_repr = {
            "type": self.encode(type(obj)),
            "repr": self.encode(dict(obj)),
        }
        return {"__convert__": convert_repr}

    def _encode_iterable(self, obj) -> str:
        # encode via converting the type and the mapping converted to list
        # separately and add conversion specifier
        convert_repr = {
            "type": self.encode(type(obj)),
            "repr": self.encode(list(obj)),
        }
        return {"__convert__": convert_repr}

    def _encode_module(self, obj) -> str:
        # encode via name and module specifier
        return {"__module__": obj.__module__}

    def _encode_type(self, obj) -> str:
        type_repr = {
            "module": self.encode(obj.__module__),
            "name": self.encode(obj.__name__),
        }
        return {"__type__": type_repr}

    def _encode_function(self, obj):
        function_repr = {
            "module": self.encode(obj.__module__),
            "name": self.encode(obj.__name__),
        }
        return {"__function__": function_repr}

    def _encode_class(self, obj) -> str:
        try:
            class_repr = {
                "type": self.encode(type(obj)),
                "dict": self.encode(obj.__dict__)
            }
            return {"__class__": class_repr}
        except Exception as e:
            logging.error(e)


class Decoder:
    def __init__(self):
        super().__init__()
        self._decode_mapping = {
            "__array__": self._decode_array,
            "__convert__": self._decode_convert,
            "__module__": self._decode_module,
            "__type__": self._decode_type,
            "__function__": self._decode_function,
            "__class__": self._decode_class,
        }

    def __call__(self, obj) -> typing.Any:
        return self.decode(obj)

    def decode(self, obj) -> typing.Any:
        if isinstance(obj, (str, int, float)):
            return obj
        elif isinstance(obj, dict):
            return self._decode_dict(obj)
        elif isinstance(obj, list):
            return self._decode_list(obj)
        else:
            return obj

    def _decode_dict(self, obj) -> dict:
        for key in obj.keys():
            if key in self._decode_mapping:
                return self._decode_mapping[key](obj[key])
            else:
                obj[key] = self.decode(obj[key])
        return obj

    def _decode_list(self, obj) -> list:
        return [self.decode(_i) for _i in obj]

    def _decode_array(self, obj) -> np.ndarray:
        return np.array(self.decode(obj))

    def _decode_convert(self, obj) -> typing.Union[typing.Iterable,
                                                   typing.Mapping]:
        # decode items in dict representation
        convert_repr = self.decode(obj)
        # create new object
        return convert_repr["type"](convert_repr["repr"])

    def _decode_module(self, obj) -> types.ModuleType:
        return importlib.import_module(self.decode(obj))

    def _decode_type(self, obj) -> typing.Any:
        # decode items in dict representation
        type_repr = self.decode(obj)
        return getattr(importlib.import_module(type_repr["module"]),
                       type_repr["name"])

    def _decode_function(self, obj) -> typing.Union[types.FunctionType,
                                                    types.BuiltinFunctionType]:
        # decode items in dict representation
        function_repr = self.decode(obj)
        return getattr(importlib.import_module(function_repr["module"]),
                       function_repr["name"])

    def _decode_class(self, obj) -> typing.Any:
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
