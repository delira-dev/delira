import importlib
import types
import collections
import inspect
import numpy as np
import logging


# ToDo: If pickling whole torch.nn.Modules (and not only state-dicts),
#  PyTorch checks for code consistency by saving the code. Should we do this
#  too, or should we state in the docs, that the behavior may be different if
#  the code changed?
class Encoder:
    def __init__(self, sep=";"):
        self._sep = sep

    def encode(self, obj):
        if isinstance(obj, str):
            return obj
        # for mappings: encode each key and each value separately and then
        # enocde the whole object to get the type
        elif isinstance(obj, collections.Mapping):
            return self._encode(type(obj)({self.encode(k): self.encode(v)
                                           for k, v in obj.items()}))
        # for iterables. encode each element separately then encode the whole
        # iterable to get the type
        elif isinstance(obj, collections.Iterable):
            return self._encode(type(obj)([self.encode(_obj)
                                           for _obj in obj]))
        else:
            return self._encode(obj)

    def _encode(self, obj):

        # if tuple: add explicit tuple specifier
        if isinstance(obj, tuple):
            return "__tuple__({})".format(obj)
        # if list: add explicit list specifier
        elif isinstance(obj, list):
            return "__list__({})".format(obj)
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
            return "__dict__({})".format(obj)
        # encode via encoding the type and the mapping converted to dict
        # separately and add a conversion specifier
        elif isinstance(obj, collections.Mapping):
            return "__convert__({}{}{})".format(self.encode(type(obj)),
                                                self._sep,
                                                self.encode(dict(obj)))
        # encode via converting the type and the mapping converted to list
        # separately and add conversion specifier
        elif isinstance(obj, collections.Iterable):
            return "__convert__({}{}{})".format(self.encode(type(obj)),
                                                self._sep,
                                                self.encode(list(obj)))

        # encode via name and module specifier
        elif isinstance(obj, types.ModuleType):
            return "__module__({})".format(obj.__module__)

        # use both ways to determine functions here
        # (the second uglier one serves as fallback here in case inspect
        # does not cover all cases)
        elif inspect.isclass(obj) or type(obj) == type:
            return "__type__({}{}{})".format(obj.__module__,
                                             self._sep,
                                             obj.__name__)
        elif isinstance(obj, (types.BuiltinFunctionType, types.FunctionType)):
            return "__function__({}{}{})".format(obj.__module__,
                                                 self._sep,
                                                 obj.__name__)

        else:
            try:
                return "__class__({}{}{})".format(self.encode(type(obj)),
                                                  self._sep,
                                                  self.encode(obj.__dict__))
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

        if obj.startswith("__tuple__"):
            return self.convert(self.decode(obj[10:-1]), tuple)
        elif obj.startswith("__list__"):
            return self.convert(self.decode(obj[9:-1]), list)
        elif obj.startswith("__array__"):
            return self.convert(self.decode(obj[10: -1]), np.array)
        elif obj.startswith("__dict__"):
            return self.convert(self.decode(obj[9:-1]), dict)
        elif obj.startswith("__convert__"):
            dtype, items = obj[12:-1].split(self._sep, 1)
            return self.convert(self.decode(items), self.decode(dtype))
        elif obj.startswith("__module__"):
            return importlib.import_module(obj[11:-1])
        elif obj.startswith("__type__"):
            module, name = obj[9:-1].split(self._sep, 1)
            return getattr(importlib.import_module(module), name)
        elif obj.startswith("__function__"):
            module, name = obj[13:-1].split(self._sep, 1)
            return getattr(importlib.import_module(module), name)
        elif obj.startswith("__class__"):

            cls_type, cls_dict = obj[10:-1].split(self._sep, 1)

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

    @staticmethod
    def convert(obj, dtype):
        return dtype(obj)

    def __call__(self, obj):
        return self.decode(obj)
