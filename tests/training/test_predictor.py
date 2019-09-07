from ..utils import check_for_no_backend
from delira.training import Predictor
import unittest
import numpy as np
from batchgenerators.transforms import RangeTransform, \
    ContrastAugmentationTransform, AbstractTransform


class MakeValueTrafo(AbstractTransform):
    def __init__(self, value):
        self._value = value

    def __call__(self, **data_dict):
        for k, v in data_dict.items():
            v = np.ones_like(v) * self._value
            data_dict[k] = v

        return data_dict


def network_dummy_fn(x):
    return {"pred": x}


def dummy_reduce_fn(x):
    return x


class TestPredictor(unittest.TestCase):
    @unittest.skipUnless(
        check_for_no_backend(),
        "Test should only be executed if no backend is installed")
    def test_tta_one_trafo(self):

        self._predictor = Predictor(model=network_dummy_fn,
                                    key_mapping={"x": "data"},
                                    tta_transforms=(
                                        RangeTransform(),
                                        ContrastAugmentationTransform()
                                    ),
                                    tta_reduce_fn=dummy_reduce_fn)

        preds = self._predictor({"data": np.random.rand(4, 3, 32, 32)})

        # check if 3 different predictions were made
        # (1 per transform + original)
        self.assertTupleEqual(preds["pred"].shape, (3, 4, 3, 32, 32))
        self.assertFalse(all((preds["pred"] == _pred).all()
                             for _pred in preds["pred"]))

    def test_tta_back_trafo(self):
        self._predictor = Predictor(model=network_dummy_fn,
                                    key_mapping={"x": "data"},
                                    tta_transforms=(
                                        MakeValueTrafo(1.),
                                    ),
                                    tta_inverse_transforms=(
                                        MakeValueTrafo(0.),
                                    ),
                                    tta_reduce_fn=dummy_reduce_fn)

        preds = self._predictor({"data": np.random.rand(4, 3, 32, 32)})

        # check for shape (1 per transform + original)
        self.assertTupleEqual(preds["pred"].shape, (2, 4, 3, 32, 32))

        # Check if all values are zero (which means,
        # the inverse trafo worked correctly)
        self.assertFalse(preds["pred"][1:].any())


if __name__ == '__main__':
    unittest.main()
