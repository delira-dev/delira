import unittest
import numpy as np

from delira.utils.dict_reductions import possible_reductions, \
    flatten_dict, unflatten_dict, reduce_dict, get_reduction


class TestDictReductions(unittest.TestCase):
    def setUp(self) -> None:
        self._reduce_sequence = [2, 3, 4, 5, 6]

        self._test_dict = {
            "a": self._reduce_sequence,
            "b": {
                "c": self._reduce_sequence
            },
            "d": {
                "e": {
                    "f": self._reduce_sequence
                }
            }
        }

        self._flattened_test_dict = {
            "a": self._reduce_sequence,
            "b.c": self._reduce_sequence,
            "d.e.f": self._reduce_sequence
        }

        self._reduction_results = {"max": max(self._reduce_sequence),
                                   "min": min(self._reduce_sequence),
                                   "mean": np.mean(self._reduce_sequence),
                                   "median": np.median(self._reduce_sequence),
                                   "first": self._reduce_sequence[0],
                                   "last": self._reduce_sequence[-1]}

        self._reduce_dicts = []
        for i in self._reduce_sequence:
            self._reduce_dicts.append(
                {
                    "a": i,
                    "b": {
                        "c": i
                    },
                    "d": {
                        "e": {
                            "f": i
                        }
                    }

                }
            )

    def test_dict_flatten(self):
        result_dict = flatten_dict(self._test_dict, parent_key='', sep=".")
        self.assertDictEqual(result_dict, self._flattened_test_dict)

    def test_dict_unflatten(self):
        result_dict = unflatten_dict(self._flattened_test_dict, sep=".")
        self.assertDictEqual(result_dict, self._test_dict)

    def test_dict_flatten_unflatten(self):
        result_dict = unflatten_dict(flatten_dict(self._test_dict,
                                                  parent_key='', sep="."),
                                     sep=".")

        self.assertDictEqual(result_dict, self._test_dict)

    def test_reduction_fuctions(self):
        for key in possible_reductions():
            with self.subTest(reduce_type=key):
                result = get_reduction(key)(self._reduce_sequence)

                # convert array to scalar if necessary
                if isinstance(result, np.ndarray):
                    result = result.item()

                self.assertEquals(result, self._reduction_results[key])

    def test_reduce_dict(self):
        for key in possible_reductions():
            with self.subTest(reduce_type=key):
                result_dict = reduce_dict(self._reduce_dicts,
                                          get_reduction(key))

                target_dict = {
                    "a": self._reduction_results[key],
                    "b": {
                        "c": self._reduction_results[key]
                    },
                    "d": {
                        "e": {
                            "f": self._reduction_results[key]
                        }
                    }

                }

                self.assertDictEqual(result_dict, target_dict)


if __name__ == '__main__':
    unittest.main()
