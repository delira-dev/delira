import unittest
from delira.models import AbstractNetwork
from delira.training import Predictor
import numpy as np
from delira.utils.config import LookupConfig


class DummyNetwork(AbstractNetwork):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return {"pred": x}

    @staticmethod
    def prepare_batch(batch: dict, input_device=None, output_device=None):
        return batch


class PredictorTest(unittest.TestCase):
    def test_prepare_metric_calc(self):
        preds = {"pred": np.random.rand(3, 4),
                 "nested": {"a": np.random.rand(3), "b": np.random.rand(3)}}

        batch_dict = {"data": np.random.rand(3, 4),
                      "label": np.random.rand(3, 4)}

        predictor = Predictor(DummyNetwork(), {"x": "data"})

        preds_batch = predictor.prepare_metric_calc(batch_dict, preds)
        self.assertIsInstance(preds_batch, LookupConfig)

        self.assertListEqual(sorted(tuple(preds_batch.keys())),
                             sorted(("data", "label", "pred", "nested")))
        self.assertIsInstance(preds_batch["nested"], LookupConfig)


if __name__ == '__main__':
    unittest.main()
