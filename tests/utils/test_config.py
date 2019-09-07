import unittest

from delira.utils.config import Config, LookupConfig, DeliraConfig
#TODO: skips


class ConfigTest(unittest.TestCase):
    def test_config_access(self):
        # initialization from dict
        example_dict = {"shallowStr": "a", "shallowNum": 1,
                        "deep": {"deepStr": "b", "deepNum": "2"}}
        cf = Config.create_from_dict(example_dict)
        self.assertEqual(cf["shallowStr"], example_dict["shallowStr"])
        self.assertEqual(cf["shallowNum"], example_dict["shallowNum"])

        # check if parameters were written correctly
        self.assertEqual(cf["deep"]["deepStr"],
                         example_dict["deep"]["deepStr"])
        self.assertEqual(cf["deep"]["deepNum"],
                         example_dict["deep"]["deepNum"])

        # check deep acces with operators
        self.assertEqual(cf["deep.deepStr"], example_dict["deep"]["deepStr"])
        self.assertEqual(cf.deep.deepNum, example_dict["deep"]["deepNum"])

        # empty initialization
        cf = Config()

        # set shallow attributes
        cf.shallowString = "string"
        cf.shallowNum = 1
        cf.deep = {}
        cf.deep.string = "deepString"
        cf.deep.num = "deepNum"

        cf["shallowString2"] = "string2"
        cf["shallowNum2"] = 2
        cf["deep.string2"] = "deepString2"
        cf["deep.num2"] = "deepNum2"

        # check if parameters were written correctly
        self.assertEqual(cf["shallowString"], "string")
        self.assertEqual(cf["shallowNum"], 1)
        self.assertEqual(cf["deep.string"], "deepString")
        self.assertEqual(cf["deep.num"], "deepNum")

        self.assertEqual(cf["shallowString2"], "string2")
        self.assertEqual(cf["shallowNum2"], 2)
        self.assertEqual(cf["deep.string2"], "deepString2")
        self.assertEqual(cf["deep.num2"], "deepNum2")

        # check contains operator
        self.assertTrue("shallowString" in cf)
        self.assertTrue("shallowString2" in cf)
        self.assertTrue("deep.string" in cf)
        self.assertTrue("deep.string2" in cf)

    def test_save_get(self):
        pass

    def test_update(self):
        pass

    def test_dump_load(self):
        pass

    def test_copy(self):
        pass

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
