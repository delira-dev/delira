import unittest
import os
import copy

from delira.utils.config import Config, LookupConfig, DeliraConfig
# TODO: skips


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.config_cls = Config
        self.example_dict = {
            "shallowStr": "a", "shallowNum": 1,
            "deep": {"deepStr": "b", "deepNum": 2},
            "nestedListOrig": [{"dictList": [1, 2, 3]}],
        }
        self.update_dict = {
            "deep": {"deepStr": "c"}, "shallowNew": 3,
            "deepNew": {"newNum": 4},
            "nestedList": [{"dictList": [1, 2, 3]}],
            "nestedList2": [{"dictList": [1, 2, 3]}],
        }

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
        self.assertTrue(isinstance(cf["deepNew"], self.config_cls))

        # check for shallow copy
        cf["nestedList"][0]["dictList"][0] = 10
        self.assertEqual(self.update_dict["nestedList"][0]["dictList"][0],
                         cf["nestedList"][0]["dictList"][0])

        # check for deepcopy
        cf.update(self.update_dict, overwrite=True, deepcopy=True)
        cf["nestedList2"][0]["dictList"][0] = 10
        self.assertNotEqual(self.update_dict["nestedList2"][0]["dictList"][0],
                            cf["nestedList2"][0]["dictList"][0])

        print("Hello World")

    def test_dump_and_load(self):
        cf = self.config_cls.create_from_dict(self.example_dict)

        # check dump
        cf.dump(os.path.join(".", "test_config.json"))

        # check load
        cf_loaded = self.config_cls.load(os.path.join(".", "test_config.json"))
        self.assertDictEqual(cf, cf_loaded)

        # check dump
        cf_string = cf.dumps()

        # check load
        cf_loaded = self.config_cls.loads(cf_string)
        self.assertDictEqual(cf, cf_loaded)

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

    def test_logging_as_string(self):
        pass

    def test_logging_as_hyperparameter(self):
        pass


class LookupConfigTest(ConfigTest):
    def test_1(self):
        pass


class DeliraConfigTest(LookupConfigTest):
    def test_1(self):
        pass
