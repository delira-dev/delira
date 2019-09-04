import importlib
import types
import collections
import inspect
import numpy as np
import logging
import json
import typing


class Encoder:
    def __call__(self, obj) -> str:
        return self.encode(obj)

    def encode(self, obj) -> str:
        if obj is None:
            return self._encode_none(obj)
        elif isinstance(obj, tuple):
            return self._encode_tuple(obj)
        elif isinstance(obj, list):
            return self._encode_list(obj)
        elif isinstance(obj, np.ndarray):
            return self._encode_array(obj)
        elif isinstance(obj, (str, int, float)):
            # if object is already a type, which can be decoded directly:
            # return as is
            return obj
        elif isinstance(obj, dict):
            return self._encode_dict(obj)
        elif isinstance(obj, collections.Mapping):
            return self._encode_mapping(obj)
        elif isinstance(obj, collections.Iterable):
            return self._encode_iterable(obj)
        elif isinstance(obj, types.ModuleType):
            return self._encode_module(obj)
        elif inspect.isclass(obj) or type(obj) == type:
            # use both ways to determine functions here
            # (the second uglier one serves as fallback here in case inspect
            # does not cover all cases)
            return self._encode_type(obj)
        elif isinstance(obj, (types.BuiltinFunctionType, types.FunctionType)):
            return self._encode_function(obj)
        else:
            return self._encode_class(obj)

    def _encode_none(self, obj) -> str:
        return "__none__"

    def _encode_tuple(self, obj) -> str:
        # if tuple: add explicit tuple specifier
        return "__tuple__({})".format(self.encode(list(obj)))

    def _encode_list(self, obj) -> str:
        # if list: add explicit list specifier
        list_repr = json.dumps([self.encode(i) for i in obj])
        return "__list__({})".format(list_repr)

    def _encode_array(self, obj) -> str:
        # # if numpy array: add explicit array specifier
        # use tolist instead of tostring here (even though this requires
        # additional encoding steps and increases memory usage), since tolist
        # retains the shape and tostring doesn't
        return "__array__({})".format(self.encode(obj.tolist()))

    def _encode_dict(self, obj) -> str:
        # if dict: add specific dict specifier
        encoded_dict = {
            _key: self.encode(_item) for _key, _item in obj.items()}
        return "__dict__({})".format(json.dumps(encoded_dict))

    def _encode_mapping(self, obj) -> str:
        # encode via encoding the type and the mapping converted to dict
        # separately and add a conversion specifier
        convert_repr = {
            "type": self.encode(type(obj)),
            "repr": self.encode(dict(obj)),
        }
        return "__convert__({})".format(self.encode(convert_repr))

    def _encode_iterable(self, obj) -> str:
        # encode via converting the type and the mapping converted to list
        # separately and add conversion specifier
        convert_repr = {
            "type": self.encode(type(obj)),
            "repr": self.encode(list(obj)),
        }
        return "__convert__({})".format(self.encode(convert_repr))

    def _encode_module(self, obj) -> str:
        # encode via name and module specifier
        return "__module__({})".format(self.encode(obj.__module__))

    def _encode_type(self, obj) -> str:
        type_repr = {
            "module": self.encode(obj.__module__),
            "name": self.encode(obj.__name__),
        }
        return "__type__({})".format(self.encode(type_repr))

    def _encode_function(self, obj):
        function_repr = {
            "module": self.encode(obj.__module__),
            "name": self.encode(obj.__name__),
        }
        return "__function__({})".format(self.encode(function_repr))

    def _encode_class(self, obj) -> str:
        try:
            class_repr = {
                "type": self.encode(type(obj)),
                "dict": self.encode(obj.__dict__)
            }
            return "__class__({})".format(self.encode(class_repr))
        except Exception as e:
            logging.error(e)


class Decoder:
    def __call__(self, obj) -> typing.Any:
        return self.decode(obj)

    def decode(self, obj) -> typing.Any:
        # if object isn't a string: return the object as is
        if not isinstance(obj, str):
            return obj

        if obj == "__none__":
            return self._decode_none(obj)
        elif obj.startswith("__tuple__"):
            return self._decode_tuple(obj)
        elif obj.startswith("__list__"):
            return self._decode_list(obj)
        elif obj.startswith("__array__"):
            return self._decode_array(obj)
        elif obj.startswith("__dict__"):
            return self._decode_dict(obj)
        elif obj.startswith("__convert__"):
            return self._decode_convert(obj)
        elif obj.startswith("__module__"):
            return self._decode_module(obj)
        elif obj.startswith("__type__"):
            return self._decode_type(obj)
        elif obj.startswith("__function__"):
            return self._decode_function(obj)
        elif obj.startswith("__class__"):
            return self._decode_class(obj)
        else:
            return obj

    def _decode_none(self, obj) -> None:
        return None

    def _decode_tuple(self, obj) -> tuple:
        return self.decode(obj[10:-1])

    def _decode_list(self, obj) -> list:
        list_repr = json.loads(obj[9:-1])
        return [self.decode(_i) for _i in list_repr]

    def _decode_array(self, obj) -> np.ndarray:
        return np.array(self.decode(obj[10: -1]))

    def _decode_dict(self, obj) -> dict:
        obj = json.loads(obj[9:-1])
        for k, v in obj.items():
            obj[self.decode(k)] = self.decode(v)
        return dict(obj)

    def _decode_convert(self, obj) -> typing.Union[typing.Iterable, typing.Mapping]:
        convert_repr = self.decode(obj[12:-1])
        return convert_repr["type"](convert_repr["repr"])

    def _decode_module(self, obj) -> types.ModuleType:
        return importlib.import_module(self.decode(obj[11:-1]))

    def _decode_type(self, obj) -> typing.Any:
        type_repr = self.decode(obj[9:-1])
        return getattr(importlib.import_module(type_repr["module"]),
                       type_repr["name"])

    def _decode_function(self, obj) -> typing.Union[types.FunctionType,
                                                    types.BuiltinFunctionType]:
        function_repr = self.decode(obj[13:-1])
        return getattr(importlib.import_module(function_repr["module"]),
                       function_repr["name"])

    def _decode_class(self, obj) -> typing.Any:
        class_repr = self.decode(obj[10:-1])
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
