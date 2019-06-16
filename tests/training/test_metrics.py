import numpy as np
from sklearn.metrics import accuracy_score
import unittest

from delira.training.metrics import SklearnClassificationMetric, \
    SklearnAccuracyScore, AurocMetric

from ..utils import check_for_no_backend


class TestMetrics(unittest.TestCase):
    @unittest.skipUnless(
        check_for_no_backend(),
        "Test should only be executed "
        "if no backend is specified")
    def test_sklearn_classification_metric(self):
        """
        Test metric wrapper for sklearn metrics
        """
        target = np.array([1, 1, 1, 1, 1])
        pred = np.array([0, 1, 0, 1, 0])
        dummy_fn = accuracy_score

        metric_wrapped = SklearnClassificationMetric(dummy_fn,
                                                     pred_logits=False,
                                                     gt_logits=False)
        wrapped_score = metric_wrapped(target, pred)
        self.assertLess(np.abs(wrapped_score - 0.4), 1e-8)

        metric_ac = SklearnAccuracyScore(gt_logits=False, pred_logits=False)
        score = metric_ac(target, pred)
        self.assertLess(np.abs(score - 0.4), 1e-8)

    @unittest.skipUnless(
        check_for_no_backend(),
        "Test should only be executed "
        "if no backend is specified")
    def test_auroc_metric(self):
        """
        Test auroc metric
        """
        pred = np.array([1, 1, 1, 1])
        target = np.array([1, 0, 1, 0])

        metric_auc = AurocMetric()
        score_auc = metric_auc(target, pred)
        self.assertEqual(score_auc, 0.5)


if __name__ == '__main__':
    unittest.main()
