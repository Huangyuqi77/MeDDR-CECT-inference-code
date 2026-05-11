"""Label-compatible evaluation helpers.

External label space may lack some classes (e.g. no CHCC). These helpers
compute metrics only on the compatible subset to avoid misleading results.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)


def stage1_external_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Binary stage-1 metrics: benign vs malignant."""
    out = {}
    out["auc"] = float(roc_auc_score(y_true, y_prob))
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["sensitivity"] = float(recall_score(y_true, y_pred, pos_label=1))
    out["specificity"] = float(recall_score(y_true, y_pred, pos_label=0))
    out["precision"] = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, pos_label=1))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out["cm"] = cm.tolist()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    out["roc_fpr"] = fpr.tolist()
    out["roc_tpr"] = tpr.tolist()
    return out


def stage2_hcc_vs_icc_metrics(y_true_4clf: np.ndarray, proba_3clf: np.ndarray,
                               class_order: Sequence[int] = (0, 1, 2)) -> dict:
    """Stage-2 external metrics restricted to HCC (0) vs ICC (1) only.

    *proba_3clf* is the 3-class probability matrix in column order
    HCC, ICC, CHCC. We reduce it to a binary HCC-vs-ICC problem by
    normalizing the first two columns.
    """
    mask = np.isin(y_true_4clf, [0, 1])
    y_true = y_true_4clf[mask]
    P = proba_3clf[mask][:, :2]
    P = P / np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)
    # ICC=1 is the "positive" class for consistency
    y_prob_icc = P[:, 1]
    y_pred = (P[:, 1] > P[:, 0]).astype(int)  # 0=HCC, 1=ICC

    out = {"n": int(mask.sum())}
    if out["n"] < 2 or len(np.unique(y_true)) < 2:
        out["error"] = "not enough classes for binary AUC"
        return out
    out["auc_icc_vs_hcc"] = float(roc_auc_score(y_true, y_prob_icc))
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["sensitivity_icc"] = float(recall_score(y_true, y_pred, pos_label=1))
    out["specificity_icc"] = float(recall_score(y_true, y_pred, pos_label=0))
    out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
    out["cm"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return out


def cascade_compatible_metrics(
    y_true_4clf: np.ndarray,
    pred_4clf: np.ndarray,
    compatible_classes: Sequence[int] = (0, 1, 3),
) -> dict:
    """Cascade metrics restricted to classes present externally."""
    mask = np.isin(y_true_4clf, list(compatible_classes))
    y_true = y_true_4clf[mask]
    y_pred = pred_4clf[mask]
    out = {"n_compatible": int(mask.sum())}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["f1_macro"] = float(f1_score(y_true, y_pred, labels=list(compatible_classes),
                                     average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(compatible_classes))
    out["cm"] = cm.tolist()
    out["labels"] = list(compatible_classes)
    return out
