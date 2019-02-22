from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, \
    fbeta_score, hamming_loss, jaccard_similarity_score, log_loss, \
    matthews_corrcoef, precision_score, recall_score, zero_one_loss
from ..utils.decorators import make_deprecated

import trixi
import numpy as np

from delira import get_backends


class SklearnClassificationMetric(object):
    def __init__(self, score_fn):
        self._score_fn = score_fn

    def __call__(self, y_true, y_pred, **kwargs):
        arrays = {"y_true": y_true, "y_pred": y_pred}

        for k, v in arrays.items():
            arrays[k] = v.reshape(v.shape[0], -1).astype(np.uint8)

        return self._score_fn(**arrays, **kwargs)


SklearnAccuracyScore = SklearnClassificationMetric(accuracy_score)
SklearnBalancedAccuracyScore = SklearnClassificationMetric(
    balanced_accuracy_score)
SklearnF1Score = SklearnClassificationMetric(f1_score)
SklearnFBetaScore = SklearnClassificationMetric(fbeta_score)
SklearnHammingLoss = SklearnClassificationMetric(hamming_loss)
SklearnJaccardSimilarityScore = SklearnClassificationMetric(
    jaccard_similarity_score)
SklearnLogLoss = SklearnClassificationMetric(log_loss)
SklearnMatthewsCorrCoeff = SklearnClassificationMetric(matthews_corrcoef)
SklearnPrecisionScore = SklearnClassificationMetric(precision_score)
SklearnRecallScore = SklearnClassificationMetric(recall_score)
SklearnZeroOneLoss = SklearnClassificationMetric(zero_one_loss)
