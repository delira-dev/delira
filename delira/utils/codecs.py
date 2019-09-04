import importlib
import types
import collections
import inspect
import numpy as np
import logging
import json


# ToDo: If pickling whole torch.nn.Modules (and not only state-dicts),
#  PyTorch checks for code consistency by saving the code. Should we do this
#  too, or should we state in the docs, that the behavior may be different if
#  the code changed?
class Encoder:
    def __init__(self, sep=";"):
        self._sep = sep

    def encode(self, obj):
        if obj is None:
            return "__none__"
        # if tuple: add explicit tuple specifier
        elif isinstance(obj, tuple):
            return "__tuple__({})".format(self.encode(list(obj)))

        # if list: add explicit list specifier
        elif isinstance(obj, list):
            list_repr = json.dumps([self.encode(i) for i in obj])
            return "__list__({})".format(list_repr)

        # # if numpy array: add explicit array specifier
        # use tolist instead of tostring here (even though this requires
        # additional encoding steps and increases memory usage), since tolist
        # retains the shape and tostring doesn't
        elif isinstance(obj, np.ndarray):
            return "__array__({})".format(self.encode(obj.tolist()))

        # if object is already a type, which can be decoded directly:
        # return as is
        elif isinstance(obj, (str, int, float)):
            return obj

        # if dict: add specific dict specifier
        elif isinstance(obj, dict):
            encoded_dict = {
                _key: self.encode(_item) for _key, _item in obj.items()}
            return "__dict__({})".format(json.dumps(encoded_dict))

        # encode via encoding the type and the mapping converted to dict
        # separately and add a conversion specifier
        elif isinstance(obj, collections.Mapping):
            convert_repr = {
                "type": self.encode(type(obj)),
                "repr": self.encode(dict(obj)),
            }
            return "__convert__({})".format(self.encode(convert_repr))

        # encode via converting the type and the mapping converted to list
        # separately and add conversion specifier
        elif isinstance(obj, collections.Iterable):
            convert_repr = {
                "type": self.encode(type(obj)),
                "repr": self.encode(list(obj)),
            }
            return "__convert__({})".format(self.encode(convert_repr))

        # encode via name and module specifier
        elif isinstance(obj, types.ModuleType):
            return "__module__({})".format(self.encode(obj.__module__))

        # use both ways to determine functions here
        # (the second uglier one serves as fallback here in case inspect
        # does not cover all cases)
        elif inspect.isclass(obj) or type(obj) == type:
            type_repr = {
                "module": self.encode(obj.__module__),
                "name": self.encode(obj.__name__),
            }
            return "__type__({})".format(self.encode(type_repr))

        elif isinstance(obj, (types.BuiltinFunctionType, types.FunctionType)):
            function_repr = {
                "module": self.encode(obj.__module__),
                "name": self.encode(obj.__name__),
            }
            return "__function__({})".format(self.encode(function_repr))

        else:
            try:
                class_repr = {
                    "type": self.encode(type(obj)),
                    "dict": self.encode(obj.__dict__)
                }
                return "__class__({})".format(self.encode(class_repr))
            except Exception as e:
                logging.error(e)

    def __call__(self, obj):
        return self.encode(obj)


class Decoder:
    def __init__(self, sep=";"):
        self._sep = sep

    def decode(self, obj):
        # if object isn't a string: return the object as is
        if not isinstance(obj, str):
            return obj
        if obj == "__none__":
            return None
        elif obj.startswith("__tuple__"):
            return tuple(self.decode(obj[10:-1]))
        elif obj.startswith("__list__"):
            list_repr = json.loads(obj[9:-1])
            return [self.decode(_i) for _i in list_repr]
        elif obj.startswith("__array__"):
            return np.array(self.decode(obj[10: -1]))
        elif obj.startswith("__dict__"):
            obj = json.loads(obj[9:-1])
            for k, v in obj.items():
                obj[self.decode(k)] = self.decode(v)
            return dict(obj)
        elif obj.startswith("__convert__"):
            convert_repr = self.decode(obj[12:-1])
            return convert_repr["type"](convert_repr["repr"])
        elif obj.startswith("__module__"):
            return importlib.import_module(self.decode(obj[11:-1]))
        elif obj.startswith("__type__"):
            type_repr = self.decode(obj[9:-1])
            return getattr(importlib.import_module(type_repr["module"]),
                           type_repr["name"])
        elif obj.startswith("__function__"):
            function_repr = self.decode(obj[13:-1])
            return getattr(importlib.import_module(function_repr["module"]),
                           function_repr["name"])
        elif obj.startswith("__class__"):
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

        else:
            return obj

    def __call__(self, obj):
        return self.decode(obj)
