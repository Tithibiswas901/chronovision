"""
metrics.py
----------
Evaluation metrics for binary change maps, comparing a predicted map
against a ground-truth mask. All inputs are treated as binary: any
nonzero pixel is "changed" (positive class), zero is "no change".

Metrics computed:
    - Precision
    - Recall
    - F1 score
    - IoU (Jaccard index)
    - Overall Accuracy (OA)
    - Cohen's Kappa

All ratios are guarded against division by zero and return 0.0 in that
case (e.g. a method that predicts no change at all, or a sample with no
ground-truth change).
"""

import numpy as np


def _safe_div(numerator, denominator):
    """Return numerator / denominator, or 0.0 if denominator is zero."""
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _confusion_counts(pred, gt):
    """Compute TP, FP, FN, TN counts between two binary maps.

    Parameters
    ----------
    pred, gt : np.ndarray
        Arrays of identical shape; any nonzero value is treated as the
        positive ("changed") class.

    Returns
    -------
    tuple of int
        (tp, fp, fn, tn)
    """
    pred_bin = pred.astype(bool)
    gt_bin = gt.astype(bool)

    tp = int(np.sum(pred_bin & gt_bin))
    fp = int(np.sum(pred_bin & ~gt_bin))
    fn = int(np.sum(~pred_bin & gt_bin))
    tn = int(np.sum(~pred_bin & ~gt_bin))
    return tp, fp, fn, tn


def compute_metrics(pred, gt):
    """Compute Precision, Recall, F1, IoU, Overall Accuracy and Kappa.

    Parameters
    ----------
    pred : np.ndarray
        Predicted binary change map (any shape; nonzero = changed).
    gt : np.ndarray
        Ground-truth binary mask, same shape as `pred`.

    Returns
    -------
    dict
        {'precision', 'recall', 'f1', 'iou', 'accuracy', 'kappa'}, each a
        float in [0, 1] (kappa can in principle dip slightly below 0,
        though that is rare for a reasonable detector).
    """
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    tp, fp, fn, tn = _confusion_counts(pred, gt)
    total = tp + fp + fn + tn

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    iou = _safe_div(tp, tp + fp + fn)
    accuracy = _safe_div(tp + tn, total)

    # Cohen's Kappa: agreement beyond what would be expected by chance.
    # pe = probability of chance agreement, derived from the marginal
    # distributions of predicted / ground-truth positives and negatives.
    if total == 0:
        kappa = 0.0
    else:
        pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / float(total ** 2)
        kappa = _safe_div(accuracy - pe, 1 - pe)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy,
        "kappa": kappa,
    }
