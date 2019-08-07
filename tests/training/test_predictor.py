from tests.utils import check_for_no_backend
from delira.training import Predictor
import unittest
import numpy as np
from batchgenerators.transforms import RangeTransform, \
    ContrastAugmentationTransform


def network_dummy_fn(x):
    return {"pred": x}


def dummy_reduce_fn(x):
    return x


class TestPredictor(unittest.TestCase):
    @unittest.skipUnless(
        check_for_no_backend(),
        "Test should only be executed if no backend is installed")
    def test_tta(self):

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
