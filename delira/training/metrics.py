from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, \
    fbeta_score, hamming_loss, jaccard_similarity_score, log_loss, \
    matthews_corrcoef, precision_score, recall_score, zero_one_loss, \
    roc_auc_score
from sklearn.preprocessing import label_binarize
from ..utils.decorators import make_deprecated

import trixi
import numpy as np

from delira import get_backends


class SklearnClassificationMetric(object):
    def __init__(self, score_fn, **kwargs):
        """
        Wraps an score function as a metric

        Parameters
        ----------
        score_fn: function
            function which should be wrapped
        kwargs:
            variable number of keyword arguments passed to score_fn function
        """
        self._score_fn = score_fn
        self.kwargs = kwargs

    def __call__(self, y_true, y_pred, **kwargs):
        """
        Compute metric with score_fn

        Parameters
        ----------
        y_true: np.ndarray
            ground truth data
        y_pred: np.ndarray
            predictions of network
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

        return self._score_fn(**arrays, **kwargs, **self.kwargs)


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


class AurocMetric(object):
    def __init__(self, classes=(0, 1), **kwargs):
        """
        Implements the auroc metric for binary and multi class classification

        Parameters
        ----------
        classes: array-like
            uniquely holds the label for each class.
        kwargs:
            variable number of keyword arguments passed to roc_auc_score
        """
        self.classes = classes
        self.kwargs = kwargs
        if len(self.classes) < 2:
            raise ValueError("At least classes 2 must exist for "
                             "classification. Only classes {} were passed to "
                             "AurocMetric.".format(classes))

    def __call__(self, y_true, y_pred, **kwargs):
        """
        Compute auroc

        Parameters
        ----------
        y_true: np.ndarray
            ground truth data with shape (N)
        y_pred: np.ndarray
            predictions of network in numpy format with shape (N, nclasses)
        kwargs:
            variable number of keyword arguments passed to roc_auc_score

        Returns
        -------
        float
            computes auc score
        """
        # binary classification
        if len(self.classes) == 2:
            # single output unit (e.g. sigmoid)
            if len(y_pred.shape) == 1 or y_pred.shape[2] == 1:
                return roc_auc_score(y_true, y_pred, **kwargs)
            # output of two units (e.g. softmax)
            elif y_pred.shape[2] == 2:
                return roc_auc_score(y_true, y_pred[:, 1], **kwargs)
            else:
                raise ValueError("Can not compute auroc metric for binary "
                                 "clases with {} predicted "
                                 "classes.".format(y_pred.shape[2]))

        # classification with multiple classes
        if len(self.classes) > 2:
            y_true_bin = label_binarize(y_true, self.classes)
            return roc_auc_score(y_true_bin, y_pred, **kwargs, **self.kwargs)
