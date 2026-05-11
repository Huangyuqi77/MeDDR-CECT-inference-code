"""Strict internal-validation score fusion helpers for V9."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score


def _as_matrix(df: pd.DataFrame, methods: list[str]) -> np.ndarray:
    return df[methods].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _nanfill_by_val(x_val: np.ndarray, x_ext: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(x_val, axis=0)
    med = np.where(np.isfinite(med), med, 0.5)
    xv = np.where(np.isfinite(x_val), x_val, med)
    xe = np.where(np.isfinite(x_ext), x_ext, med)
    return xv, xe


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y.astype(int), score.astype(float)))


def _rank_transform_fit_apply(x_val: np.ndarray, x_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out_val = np.zeros_like(x_val, dtype=float)
    out_new = np.zeros_like(x_new, dtype=float)
    n = x_val.shape[0]
    for j in range(x_val.shape[1]):
        order = np.argsort(x_val[:, j])
        ranks = np.empty(n, dtype=float)
        ranks[order] = (np.arange(n) + 1) / (n + 1)
        out_val[:, j] = ranks
        sorted_val = np.sort(x_val[:, j])
        out_new[:, j] = np.searchsorted(sorted_val, x_new[:, j], side="right") / (n + 1)
    return out_val, np.clip(out_new, 0.0, 1.0)


def _z_transform_fit_apply(x_val: np.ndarray, x_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = x_val.mean(axis=0)
    sd = x_val.std(axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    zv = (x_val - mu) / sd
    zn = (x_new - mu) / sd
    # Convert averaged z score back to a bounded probability-like score later.
    return zv, zn


def _bounded(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _nonnegative_weights(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    n_methods = x.shape[1]
    init = np.full(n_methods, 1.0 / n_methods)

    def obj(w):
        w = np.clip(w, 0, None)
        w = w / max(1e-12, w.sum())
        s = np.clip(x @ w, 1e-6, 1 - 1e-6)
        return log_loss(y, s)

    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    res = minimize(obj, init, bounds=[(0.0, 1.0)] * n_methods, constraints=cons, method="SLSQP", options={"maxiter": 500})
    w = res.x if res.success else init
    w = np.clip(w, 0, None)
    return w / max(1e-12, w.sum())


def fit_score_fusions(val_scores: pd.DataFrame, ext_scores: pd.DataFrame, methods: list[str], seed: int = 42) -> dict[str, dict]:
    """Fit all declared score-fusion rules using internal validation only."""
    methods = [m for m in methods if m in val_scores.columns and m in ext_scores.columns]
    if len(methods) < 2:
        return {}
    y_val = val_scores["y_true"].astype(int).to_numpy()
    xv, xe = _nanfill_by_val(_as_matrix(val_scores, methods), _as_matrix(ext_scores, methods))
    out: dict[str, dict] = {}

    def add(name: str, val_score: np.ndarray, ext_score: np.ndarray, weights: np.ndarray | None, detail: str):
        out[name] = {
            "method": name,
            "val_score": np.asarray(val_score, dtype=float),
            "ext_score": np.asarray(ext_score, dtype=float),
            "weights": {m: float(w) for m, w in zip(methods, weights)} if weights is not None else {},
            "val_auc": _auc(y_val, np.asarray(val_score, dtype=float)),
            "detail": detail,
        }

    uniform = np.full(len(methods), 1.0 / len(methods))
    add("F0_uniform_probability_average", xv @ uniform, xe @ uniform, uniform, "uniform probability average")

    rv, re = _rank_transform_fit_apply(xv, xe)
    add("F1_rank_normalized_average", rv @ uniform, re @ uniform, uniform, "rank transform fit on internal validation only")

    zv, ze = _z_transform_fit_apply(xv, xe)
    add("F2_zscore_normalized_average", _bounded(zv @ uniform), _bounded(ze @ uniform), uniform, "z transform fit on internal validation only")

    try:
        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=seed).fit(xv, y_val)
        val = lr.predict_proba(xv)[:, 1]
        ext = lr.predict_proba(xe)[:, 1]
        w = np.ravel(lr.coef_).astype(float)
        add("F3_holdout_logistic_stacking", val, ext, w, "logistic stacking fit on internal-validation out-of-sample branch scores")
    except Exception:
        pass

    w_nn = _nonnegative_weights(y_val, xv)
    add("F5_constrained_nonnegative_stacking", xv @ w_nn, xe @ w_nn, w_nn, "non-negative weights fit on internal validation")

    learned = w_nn
    best = None
    for lam in np.linspace(0, 1, 11):
        w = lam * learned + (1 - lam) * uniform
        w = w / max(1e-12, w.sum())
        val = xv @ w
        va = _auc(y_val, val)
        if best is None or va > best[0]:
            best = (va, lam, w, val, xe @ w)
    if best is not None:
        _, lam, w, val, ext = best
        add("F4_shrinkage_stacking", val, ext, w, f"lambda={lam:.2f}; learned weights shrunk toward uniform")

    aucs = np.asarray([max(_auc(y_val, xv[:, i]) - 0.5, 1e-6) for i in range(len(methods))], dtype=float)
    w_auc = aucs / aucs.sum()
    add("F6_val_auc_weighted_average", xv @ w_auc, xe @ w_auc, w_auc, "weights proportional to internal validation AUC-0.5")
    return out
