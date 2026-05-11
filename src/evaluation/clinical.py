"""Clinical validation helpers for strict radiomics experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def bootstrap_auc(y_true, score, n_boot: int = 2000, seed: int = 42) -> dict:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    auc = float(roc_auc_score(y, s))
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(float(roc_auc_score(y[idx], s[idx])))
    if vals:
        lo, hi = np.quantile(vals, [0.025, 0.975])
    else:
        lo = hi = np.nan
    return {"auc": auc, "auc_ci_low": float(lo), "auc_ci_high": float(hi)}


def locked_threshold_metrics(y_val, p_val, y_ext, p_ext, n_boot: int = 2000, seed: int = 42) -> dict:
    """Lock threshold by internal-val Youden J; evaluate on external."""
    yv = np.asarray(y_val).astype(int)
    sv = np.asarray(p_val, dtype=float)
    ye = np.asarray(y_ext).astype(int)
    se = np.asarray(p_ext, dtype=float)

    fpr, tpr, thr = roc_curve(yv, sv)
    idx = int(np.argmax(tpr - fpr))
    threshold = float(thr[idx])
    if not np.isfinite(threshold):
        threshold = 0.5

    def _ops(yy, ss):
        pred = (ss >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(yy, pred, labels=[0, 1]).ravel()
        sens = tp / max(1, tp + fn)
        spec = tn / max(1, tn + fp)
        ppv = tp / max(1, tp + fp)
        npv = tn / max(1, tn + fn)
        acc = (tp + tn) / max(1, tp + fp + tn + fn)
        return {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "ppv": float(ppv),
            "npv": float(npv),
            "accuracy": float(acc),
            "balanced_accuracy": float(balanced_accuracy_score(yy, pred)),
            "f1": float(f1_score(yy, pred, zero_division=0)),
        }

    out = {"threshold_locked": threshold}
    out.update(bootstrap_auc(ye, se, n_boot=n_boot, seed=seed))
    out.update(_ops(ye, se))

    rng = np.random.default_rng(seed)
    boot_keys = ["sensitivity", "specificity", "ppv", "npv", "accuracy", "balanced_accuracy", "f1"]
    boot = {k: [] for k in boot_keys}
    n = len(ye)
    for _ in range(int(n_boot)):
        bidx = rng.integers(0, n, size=n)
        if len(np.unique(ye[bidx])) < 2:
            continue
        vals = _ops(ye[bidx], se[bidx])
        for k in boot_keys:
            boot[k].append(vals[k])
    for k, vals in boot.items():
        if vals:
            lo, hi = np.quantile(vals, [0.025, 0.975])
            out[f"{k}_ci_low"] = float(lo)
            out[f"{k}_ci_high"] = float(hi)
    return out


def calibration_metrics(y_true, prob, n_bins: int = 10) -> tuple[dict, pd.DataFrame]:
    y = np.asarray(y_true).astype(int)
    s = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    brier = float(brier_score_loss(y, s))
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (s >= lo) & ((s < hi) if i < n_bins - 1 else (s <= hi))
        if not mask.any():
            rows.append({"bin_low": lo, "bin_high": hi, "n": 0, "mean_pred": np.nan, "frac_positive": np.nan})
            continue
        mp = float(s[mask].mean())
        fp = float(y[mask].mean())
        ece += (mask.sum() / len(y)) * abs(fp - mp)
        rows.append({"bin_low": lo, "bin_high": hi, "n": int(mask.sum()), "mean_pred": mp, "frac_positive": fp})
    try:
        logit = np.log(s / (1.0 - s)).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, solver="lbfgs").fit(logit, y)
        intercept = float(lr.intercept_[0])
        slope = float(lr.coef_[0, 0])
    except Exception:
        intercept = np.nan
        slope = np.nan
    return {
        "brier": brier,
        "ece": float(ece),
        "calib_intercept": intercept,
        "calib_slope": slope,
    }, pd.DataFrame(rows)


def decision_curve(y_true, prob, thresholds=None) -> pd.DataFrame:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(prob, dtype=float)
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    n = len(y)
    rows = []
    for pt in thresholds:
        pred = (s >= pt).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        nb = (tp / n) - (fp / n) * (pt / (1.0 - pt))
        tp_all = int((y == 1).sum())
        fp_all = int((y == 0).sum())
        nb_all = (tp_all / n) - (fp_all / n) * (pt / (1.0 - pt))
        rows.append({
            "threshold": float(pt),
            "net_benefit_model": float(nb),
            "net_benefit_treat_all": float(nb_all),
            "net_benefit_treat_none": 0.0,
        })
    return pd.DataFrame(rows)
