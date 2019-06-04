import unittest
from copy import copy, deepcopy

from delira.training import Parameters
from delira.utils import LookupConfig


class ParametersTest(unittest.TestCase):
    def test_parameters(self):
        def to_lookup_config(dictionary):
            tmp = LookupConfig()
            tmp.update(dictionary)
            return tmp

        test_cases = [
            (
                {"a": 1, "b": [1, 2], "c": {"d": 3, "e": 56}},
                {"f": 1, "g": {"h": {"i": {"a": 3}}}},
                {"j": 1, "k": 2},
                {},
                "e",
                56,
                "a",
                "q"
            )
        ]

        for case in test_cases:
            with self.subTest(case=case):
                fixed_model_params, variable_model_params, \
                    fixed_training_params, variable_training_params, \
                    valid_nested_key, valid_nested_value, doubled_key, \
                    invalid_key = case

                fixed_model_params = to_lookup_config(fixed_model_params)
                variable_model_params = to_lookup_config(variable_model_params)
                fixed_training_params = to_lookup_config(fixed_training_params)
                variable_training_params = to_lookup_config(
                    variable_training_params)

                params = Parameters(
                    fixed_params={
                        "model": fixed_model_params,
                        "training": fixed_training_params
                    },
                    variable_params={
                        "model": variable_model_params,
                        "training": variable_training_params
                    }
                )

                self.assertFalse(params.training_on_top)
                self.assertTrue(params.variability_on_top)
                self.assertEqual(params.fixed, to_lookup_config({
                    "model": fixed_model_params,
                    "training": fixed_training_params
                }))

                self.assertEqual(params.variable, to_lookup_config({
                    "model": variable_model_params,
                    "training": variable_training_params
                }))

                params = params.permute_training_on_top()

                self.assertFalse(params.variability_on_top)
                self.assertTrue(params.training_on_top)

                self.assertEqual(params.model, to_lookup_config({
                    "fixed": fixed_model_params,
                    "variable": variable_model_params
                }))

                self.assertEqual(params.training, to_lookup_config({
                    "fixed": fixed_training_params,
                    "variable": variable_training_params
                }))

                params_copy = params.deepcopy()
                params = params.permute_variability_on_top(
                ).permute_training_on_top()
                self.assertEqual(params_copy, params)

                self.assertEqual(params.nested_get(
                    valid_nested_key), valid_nested_value)

                with self.assertRaises(KeyError):
                    params.nested_get(doubled_key)

                with self.assertRaises(KeyError):
                    params.nested_get(invalid_key)

                self.assertEqual("default", params.nested_get(
                    invalid_key, "default"))
                self.assertEqual("default", params.nested_get(
                    invalid_key, default="default"))

                params_shallow = copy(params.permute_training_on_top())
                self.assertEqual(params.training_on_top,
                                 params_shallow.training_on_top)
                params_shallow["model.fixed.a"] += 1
                self.assertNotEqual(params_shallow["model.fixed.a"],
                                    params["model.fixed.a"])

                params_shallow2 = copy(params.permute_variability_on_top())
                self.assertEqual(params.variability_on_top,
                                 params_shallow2.variability_on_top)

                params_deep = deepcopy(params.permute_training_on_top())
                self.assertEqual(params.training_on_top,
                                 params_deep.training_on_top)
                params_deep["model.fixed.a"] += 1
                self.assertNotEqual(params_deep["model.fixed.a"],
                                    params["model.fixed.a"])

                params_deep2 = deepcopy(params.permute_variability_on_top())
                self.assertEqual(params.variability_on_top,
                                 params_deep2.variability_on_top)

                params.permute_training_on_top()
                params_deep.permute_variability_on_top()
                params.update(params_deep)
                self.assertTrue(params.training_on_top)
                self.assertTrue(params_deep.variability_on_top)

                with self.assertRaises(RuntimeError):
                    params.update({"a": 1})


if __name__ == '__main__':
    unittest.main()
