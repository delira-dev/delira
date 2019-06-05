
from sklearn.metrics import accuracy_score, balanced_accuracy_score, \
    f1_score, fbeta_score, hamming_loss, jaccard_similarity_score, log_loss, \
    matthews_corrcoef, precision_score, recall_score, zero_one_loss, \
    roc_auc_score
from sklearn.preprocessing import label_binarize

import numpy as np


class SklearnClassificationMetric(object):
    def __init__(self, score_fn, gt_logits=False, pred_logits=True, **kwargs):
        """
        Wraps an score function as a metric

        Parameters
        ----------
        score_fn : function
            function which should be wrapped
        gt_logits : bool
            whether given ``y_true`` are logits or not
        pred_logits : bool
            whether given ``y_pred`` are logits or not
        **kwargs:
            variable number of keyword arguments passed to score_fn function
        """
        self._score_fn = score_fn
        self._gt_logits = gt_logits
        self._pred_logits = pred_logits
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

        if self._gt_logits:
            y_true = np.argmax(y_true, axis=-1)

        if self._pred_logits:
            y_pred = np.argmax(y_pred, axis=-1)

        return self._score_fn(y_true=y_true, y_pred=y_pred,
                              **kwargs, **self.kwargs)


class SklearnAccuracyScore(SklearnClassificationMetric):
    """
    Accuracy Metric
    """

    def __init__(self, gt_logits=False, pred_logits=True, **kwargs):
        super().__init__(accuracy_score, gt_logits, pred_logits, **kwargs)


class SklearnBalancedAccuracyScore(SklearnClassificationMetric):
    """
    Balanced Accuracy Metric
    """

    def __init__(self, gt_logits=False, pred_logits=True, **kwargs):
        super().__init__(balanced_accuracy_score, gt_logits, pred_logits,
                         **kwargs)


class SklearnF1Score(SklearnClassificationMetric):
    """
    F1 Score
    """

    def __init__(self, gt_logits=False, pred_logits=True, **kwargs):
        super().__init__(f1_score, gt_logits, pred_logits, **kwargs)


class SklearnFBetaScore(SklearnClassificationMetric):
    """
    F-Beta Score (Generalized F1)
    """

    def __init__(self, gt_logits=False, pred_logits=True, **kwargs):
        super().__init__(fbeta_score, gt_logits, pred_logits, **kwargs)


class SklearnHammingLoss(SklearnClassificationMetric):
    """
    Hamming Loss
    """

    def __init__(self, gt_logits=False, pred_logits=True, **kwargs):
        super().__init__(hamming_loss, gt_logits, pred_logits, **kwargs)


class SklearnJaccardSimilarityScore(SklearnClassificationMetric):
    """
    Jaccard Similarity Score
    """

    def __init__(self, gt_logits=False, pred_logits=True, **kwargs):
        super().__init__(jaccard_similarity_score, gt_logits, pred_logits,
                         **kwargs)


class SklearnLogLoss(SklearnClassificationMetric):
    """
    Log Loss (NLL)
    """

    def __init__(self, gt_logits=False, pred_logits=True, **kwargs):
        super().__init__(log_loss, gt_logits, pred_logits, **kwargs)


class SklearnMatthewsCorrCoeff(SklearnClassificationMetric):
    """
    Matthews Correlation Coefficient
    """

    def __init__(self, gt_logits=False, pred_logits=True, **kwargs):
        super().__init__(matthews_corrcoef, gt_logits, pred_logits, **kwargs)


class SklearnPrecisionScore(SklearnClassificationMetric):
    """
    Precision Score
    """

    def __init__(self, gt_logits=False, pred_logits=True, **kwargs):
        super().__init__(precision_score, gt_logits, pred_logits, **kwargs)


class SklearnRecallScore(SklearnClassificationMetric):
    """
    Recall Score
    """

    def __init__(self, gt_logits=False, pred_logits=True, **kwargs):
        super().__init__(recall_score, gt_logits, pred_logits, **kwargs)


class SklearnZeroOneLoss(SklearnClassificationMetric):
    """
    Zero One Loss
    """

    def __init__(self, gt_logits=False, pred_logits=True, **kwargs):
        super().__init__(zero_one_loss, gt_logits, pred_logits, **kwargs)


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

        Raises
        ------
        ValueError
            if not at least two classes are provided
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

        Raises
        ------
        ValueError
            if two classes are given and the predictions contain more than two
            classes
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
                                 "classes with {} predicted "
                                 "classes.".format(y_pred.shape[2]))

        # classification with multiple classes
        if len(self.classes) > 2:
            y_true_bin = label_binarize(y_true, self.classes)
            return roc_auc_score(y_true_bin, y_pred, **kwargs, **self.kwargs)
