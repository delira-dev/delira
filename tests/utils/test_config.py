import unittest
import os
import sys
import copy
import argparse
from unittest.mock import patch
from delira._version import get_versions

from delira.utils.config import Config, LookupConfig, DeliraConfig
from delira.logging import Logger, TensorboardBackend, make_logger, \
    register_logger
import warnings

from . import check_for_no_backend


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.config_cls = Config
        self.example_dict = {
            "shallowStr": "a",
            "shallowNum": 1,
            "deep": {"deepStr": "b", "deepNum": 2},
            "nestedListOrig": [{"dictList": [1, 2, 3]}],
        }
        self.update_dict = {
            "deep": {"deepStr": "c"},
            "shallowNew": 3,
            "deepNew": {"newNum": 4},
            "nestedList": [{"dictList": [1, 2, 3]}],
            "nestedList2": [{"dictList": [1, 2, 3]}],
        }

        self._logger = self._setup_logger()
        register_logger(self._logger, __file__)

    def _setup_logger(self):
        return make_logger(TensorboardBackend(
            {"logdir": os.path.join(".", "runs", self._testMethodName)}
        ))

    @unittest.skipUnless(
        check_for_no_backend(),
        "Test should only be executed if no backend is specified")
    def test_config_access(self):
        # initialization from dict
        cf = self.config_cls(self.example_dict)
        self.assertEqual(cf["shallowStr"], self.example_dict["shallowStr"])
        self.assertEqual(cf["shallowNum"], self.example_dict["shallowNum"])

        # check if parameters were written correctly
        self.assertEqual(cf["deep"]["deepStr"],
                         self.example_dict["deep"]["deepStr"])
        self.assertEqual(cf["deep"]["deepNum"],
                         self.example_dict["deep"]["deepNum"])

        # check deep acces with operators
        self.assertEqual(cf["deep.deepStr"],
                         self.example_dict["deep"]["deepStr"])
        self.assertEqual(cf.deep.deepNum,
                         self.example_dict["deep"]["deepNum"])

        # empty initialization
        cf = self.config_cls()

        # set shallow attributes
        cf.shallowString = "string"
        cf.shallowNum = 1
        cf.deep = {}
        cf.deep.string = "deepString"
        cf.deep.num = 2

        cf["shallowString2"] = "string2"
        cf["shallowNum2"] = 1
        cf["deep.string2"] = "deepString2"
        cf["deep.num2"] = 2

        # check if parameters were written correctly
        self.assertEqual(cf["shallowString"], "string")
        self.assertEqual(cf["shallowNum"], 1)
        self.assertEqual(cf["deep.string"], "deepString")
        self.assertEqual(cf["deep.num"], 2)

        self.assertEqual(cf["shallowString2"], "string2")
        self.assertEqual(cf["shallowNum2"], 1)
        self.assertEqual(cf["deep.string2"], "deepString2")
        self.assertEqual(cf["deep.num2"], 2)

        # check contains operator
        self.assertTrue("shallowString" in cf)
        self.assertTrue("shallowString2" in cf)
        self.assertTrue("deep.string" in cf)
        self.assertTrue("deep.string2" in cf)

        warning_msg = ("The key 5 is not a string, but a <class 'int'>. "
                       "This may lead to unwanted behavior!")
        with self.assertWarns(RuntimeWarning, msg=warning_msg):
            cf[5] = 10

    @unittest.skipUnless(
        check_for_no_backend(),
        "Test should only be executed if no backend is specified")
    def test_config_access_with_non_existing_keys(self):
        cf = self.config_cls(self.example_dict)

        with self.assertRaises(KeyError):
            cf["unknown_key"]

        with self.assertRaises(KeyError):
            cf["shallowStr.unknown_key"]

    @unittest.skipUnless(
        check_for_no_backend(),
        "Test should only be executed if no backend is specified")
    def test_update(self):
        cf = self.config_cls.create_from_dict(self.example_dict)
        with self.assertRaises(ValueError):
            cf.update(self.update_dict)

        # update with overwrite
        cf.update(self.update_dict, overwrite=True)
        self.assertEqual(cf["deep.deepStr"],
                         self.update_dict["deep"]["deepStr"])

        # add new values
        self.assertEqual(cf["shallowNew"],
                         self.update_dict["shallowNew"])
        self.assertEqual(cf["deepNew.newNum"],
                         self.update_dict["deepNew"]["newNum"])

        # check for shallow copy
        cf["nestedList"][0]["dictList"][0] = 10
        self.assertEqual(self.update_dict["nestedList"][0]["dictList"][0],
                         cf["nestedList"][0]["dictList"][0])

        # check for deepcopy
        cf.update(self.update_dict, overwrite=True, deepcopy=True)
        cf["nestedList2"][0]["dictList"][0] = 10
        self.assertNotEqual(self.update_dict["nestedList2"][0]["dictList"][0],
                            cf["nestedList2"][0]["dictList"][0])

        # check for no error when only updating nested keys
        cf = self.config_cls.create_from_dict(self.example_dict)
        update_dict = copy.deepcopy(self.update_dict)
        update_dict["deep"].pop("deepStr")
        update_dict["deep"]["deepStr2"] = "deepStr2"
        cf.update(update_dict)
        self.assertEqual(cf["deep.deepStr2"],
                         update_dict["deep"]["deepStr2"])

    @unittest.skipUnless(
        check_for_no_backend(),
        "Test should only be executed if no backend is specified")
    def test_dump_and_load(self):
        cf = self.config_cls.create_from_dict(self.example_dict)
        path = os.path.join(".", "test_config.yaml")
        # check dump
        cf.dump(path)

        # check load
        cf_loaded = self.config_cls()
        cf_loaded.load(path)
        self.assertDictEqual(cf, cf_loaded)

        cf_loaded_file = self.config_cls.create_from_file(path)
        self.assertDictEqual(cf, cf_loaded_file)

        # check dump
        cf_string = cf.dumps()

        # check load
        cf_loaded = self.config_cls()
        cf_loaded.loads(cf_string)
        self.assertDictEqual(cf, cf_loaded)

        cf_loaded_str = self.config_cls.create_from_str(cf_string)
        self.assertDictEqual(cf, cf_loaded_str)

    @unittest.skipUnless(
        check_for_no_backend(),
        "Test should only be executed if no backend is specified")
    def test_copy(self):
        cf = self.config_cls.create_from_dict(self.example_dict)

        # check for shallow copy
        cf_shallow = copy.copy(cf)
        cf_shallow["nestedListOrig"][0]["dictList"][0] = 10
        self.assertEqual(cf["nestedListOrig"][0]["dictList"][0],
                         cf_shallow["nestedListOrig"][0]["dictList"][0])

        # check for deepcopy
        cf_deep = copy.deepcopy(cf)
        cf_deep["nestedListOrig"][0]["dictList"][0] = 20
        self.assertNotEqual(cf["nestedListOrig"][0]["dictList"][0],
                            cf_deep["nestedListOrig"][0]["dictList"][0])

    @unittest.skipUnless(
        check_for_no_backend(),
        "Test should only be executed if no backend is specified")
    def test_create_from_argparse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-p1')
        parser.add_argument('--param2')
        cf1 = self.config_cls.create_from_argparse(
            parser, args=['-p1', 'parameter1', '--param2', 'parameter2'])
        self.assertEqual(cf1['p1'], 'parameter1')
        self.assertEqual(cf1['param2'], 'parameter2')

        args = parser.parse_args(
            ['-p1', 'parameter1', '--param2', 'parameter2'])
        self.assertEqual(cf1['p1'], 'parameter1')
        self.assertEqual(cf1['param2'], 'parameter2')

    @unittest.skipUnless(
        check_for_no_backend(),
        "Test should only be executed if no backend is specified")
    def test_internal_type(self):
        cf = self.config_cls.create_from_dict(self.example_dict)
        self.assertTrue(isinstance(cf["deep"], self.config_cls))

    @unittest.skipUnless(
        check_for_no_backend(),
        "Test should only be executed if no backend is specified")
    def test_create_argparser(self):
        cf = self.config_cls.create_from_dict(self.example_dict)
        testargs = [
            '--shallowNum',
            '10',
            '--deep.deepStr',
            'check',
            '--testlist',
            'ele1',
            'ele2',
            '--setflag']
        parser = cf.create_argparser()
        known, unknown = parser.parse_known_args(testargs)
        self.assertEqual(vars(known)['shallowNum'], 10)
        self.assertEqual(vars(known)['deep.deepStr'], 'check')
        self.assertEqual(unknown, ['--testlist', 'ele1', 'ele2', '--setflag'])

    @unittest.skipUnless(
        check_for_no_backend(),
        "Test should only be executed if no backend is specified")
    def test_update_from_argparse(self):
        cf = self.config_cls.create_from_dict(self.example_dict)
        testargs = ['--shallowNum', '10',
                    '--deep.deepStr', 'check',
                    '--testlist', 'ele1', 'ele2',
                    '--setflag']
        # placeholder pyfile because argparser omits first argument from sys
        # argv
        with patch.object(sys, 'argv', ['pyfile.py'] + testargs):
            cf.update_from_argparse(add_unknown_items=True)
        self.assertEqual(cf['shallowNum'], int(testargs[1]))
        self.assertEqual(cf['deep']['deepStr'], testargs[3])
        self.assertEqual(cf['testlist'], testargs[5:7])
        self.assertEqual(cf['setflag'], True)
        with warnings.catch_warnings(record=True) as w:
            with patch.object(sys, 'argv', ['pyfile.py', '--unknown', 'arg']):
                cf.update_from_argparse(add_unknown_items=False)
        self.assertEqual(len(w), 1)


class LookupConfigTest(ConfigTest):
    def setUp(self):
        super().setUp()
        self.config_cls = LookupConfig

    @unittest.skipUnless(
        check_for_no_backend(),
        "Test should only be executed if no backend is specified")
    def test_nested_lookpup(self):
        cf = self.config_cls.create_from_dict(self.example_dict)
        self.assertEqual(cf["deep.deepStr"],
                         cf.nested_get("deep.deepStr"))
        self.assertEqual(cf["deep.deepNum"], cf.nested_get("deepNum"))

        with self.assertRaises(KeyError):
            cf.nested_get("nonExistingKey")

        cf["deepStr"] = "duplicate"
        with self.assertRaises(KeyError):
            cf.nested_get("deepStr")

        self.assertIsNone(cf.nested_get("nonExistingKey", None))
        self.assertIsNone(cf.nested_get("nonExistingKey", default=None))

        cf["nested_duplicate.deep"] = "duplicate"
        with self.assertRaises(KeyError):
            cf.nested_get("deep")

        multiple_val = cf.nested_get("deep", allow_multiple=True)

        expected_result = [{"deepStr": "b", "deepNum": 2},
                           "duplicate"]

        for val in multiple_val:
            self.assertIn(val, expected_result)
            expected_result.pop(expected_result.index(val))

        self.assertEquals(len(expected_result), 0)


class DeliraConfigTest(LookupConfigTest):
    def setUp(self):
        super().setUp()
        self.config_cls = DeliraConfig

    @unittest.skipUnless(
        check_for_no_backend(),
        "Test should only be executed if no backend is specified")
    def test_property_params(self):
        for mode in ["fixed", "variable"]:
            cf = self.config_cls.create_from_dict({})
            setattr(cf, "{}_params".format(mode),
                    {"model": {"num_classes": 3}, "training": {"epochs": 2}})

            # manual checking of values
            self.assertEqual(cf["{}_model.num_classes".format(mode)], 3)
            self.assertEqual(cf["{}_training.epochs".format(mode)], 2)

            # check getter
            params = getattr(cf, "{}_params".format(mode))
            self.assertEqual(params["model.num_classes"], 3)
            self.assertEqual(params["training.epochs"], 2)

        for mode in ["training", "model"]:
            cf = self.config_cls.create_from_dict(self.example_dict)
            setattr(cf, "{}_params".format(mode),
                    {"fixed": {"num_classes": 3}, "variable": {"epochs": 2}})

            # manual checking of values
            self.assertEqual(cf["fixed_{}.num_classes".format(mode)], 3)
            self.assertEqual(cf["variable_{}.epochs".format(mode)], 2)

            # check getter
            params = getattr(cf, "{}_params".format(mode))
            self.assertEqual(params["fixed.num_classes"], 3)
            self.assertEqual(params["variable.epochs"], 2)

    @unittest.skipUnless(
        check_for_no_backend(),
        "Test should only be executed if no backend is specified")
    def test_logging_as_string(self):
        cf = self.config_cls()
        cf.update({"augment": True})
        cf.update({"fixed_model": "fm", "fixed_training": "ft",
                   "variable_model": "vm", "variable_training": "vt"},
                  overwrite=True)

        cf_str = cf.log_as_string()
        cf_str_full = cf.log_as_string(full_config=True)

        self.assertEqual(cf_str,
                         ("__convert__:\n"
                          "  repr:\n"
                          "    _timestamp: {}\n"
                          "    fixed_model: fm\n"
                          "    fixed_training: ft\n"
                          "    variable_model: vm\n"
                          "    variable_training: vt\n"
                          "  type:\n"
                          "    __type__:\n"
                          "      module: delira.utils.config\n"
                          "      name: LookupConfig\n".format(
                              cf["_timestamp"])))

        self.assertEqual(cf_str_full,
                         ("__convert__:\n"
                          "  repr:\n"
                          "    _timestamp: {}\n"
                          "    _version: {}\n"
                          "    augment: true\n"
                          "    fixed_model: fm\n"
                          "    fixed_training: ft\n"
                          "    variable_model: vm\n"
                          "    variable_training: vt\n"
                          "  type:\n"
                          "    __type__:\n"
                          "      module: delira.utils.config\n"
                          "      name: DeliraConfig\n".format(
                              cf["_timestamp"], cf["_version"])))

    @unittest.skipUnless(
        check_for_no_backend(),
        "Test should only be executed if no backend is specified")
    def test_internal_type(self):
        cf = self.config_cls.create_from_dict(self.example_dict)
        self.assertTrue(isinstance(cf["deep"], LookupConfig))


if __name__ == '__main__':
    unittest.main()
