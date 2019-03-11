from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, \
    fbeta_score, hamming_loss, jaccard_similarity_score, log_loss, \
    matthews_corrcoef, precision_score, recall_score, zero_one_loss, \
    roc_auc_score
from sklearn.preprocessing import label_binarize
from ..utils.decorators import make_deprecated

import trixi
import numpy as np

from delira import get_backends


# TODO: check input type of y_true, y_pred
class SklearnClassificationMetric(object):
    def __init__(self, score_fn):
        """
        Wraps an score function as a metric

        Parameters
        ----------
        score_fn: function
            function which should be wrapped
        """
        self._score_fn = score_fn

    def __call__(self, y_true, y_pred, **kwargs):
        """
        Compute metric with score_fn

        Parameters
        ----------
        y_true: np.ndarray
            ground truth data
        y_pred: np.ndarray
            predicted result
        kwargs:
            variable number of keyword arguments passed to score_fn

        Returns
        -------
        float
            result from score function
        """
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


# TODO: check input format of y_true and y_pred
# TODO: Unit test
class AurocMetric(object):
    def __init__(self, classes=(0, 1), **kwargs):
        """


        Parameters
        ----------
        classes
        kwargs:
            variable number of keyword arguments passed to label_binarize
        """
        self.classes = classes
        self.kwargs = kwargs
        if len(self.classes) < 2:
            raise ValueError("At least classes 2 must exist for classification. "
                             "Only classes {} were passed to "
                             "AurocMetric.".format(classes))

    def __call__(self, y_true, y_pred, **kwargs):
        """
        Compute auroc

        Parameters
        ----------
        y_true:
        y_pred:
        kwargs:
            variable number of keyword arguments passed to roc_auc_score
        Returns
        -------

        """
        # binary classification
        if len(self.classes) == 2:
            # output of two units (e.g. softmax)
            if y_pred.shape[2] == 2:
                return roc_auc_score(y_true, y_pred[:, 1], **kwargs)
            # single output unit (e.g. sigmoid)
            elif y_pred.shape[2] == 1:
                return roc_auc_score(y_true, y_pred, **kwargs)
            else:
                raise ValueError("Can not compute auroc metric for binary "
                                 "clases with {} predicted "
                                 "classes.".format(y_pred.shape[2]))

        # classification with multiple classes
        if len(self.classes) > 2:
            y_true_bin = label_binarize(y_true, self.classes, **self.kwargs)
            return roc_auc_score(y_true_bin, y_pred, **kwargs)
