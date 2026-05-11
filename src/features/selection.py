"""Stability/generalization-aware feature selection pipeline.

All functions operate on pandas DataFrames and return both the filtered
feature list and a reporting DataFrame describing the decision per feature.

Fitting uses training data only. Validation/external transformation is
handled by simply subsetting columns.
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


# ===================================================================
# Stage A: hygiene
# ===================================================================
def missing_filter(df: pd.DataFrame, features: Sequence[str],
                   max_rate: float = 0.30) -> tuple[list[str], pd.DataFrame]:
    """Drop features whose missing rate on the training set exceeds max_rate."""
    report = []
    kept = []
    for f in features:
        col = df[f]
        miss = col.isna().mean()
        passed = miss <= max_rate
        report.append({"feature": f, "missing_rate": float(miss), "kept": bool(passed)})
        if passed:
            kept.append(f)
    return kept, pd.DataFrame(report)


def variance_filter(df: pd.DataFrame, features: Sequence[str],
                    method: str = "nzv", nzv_uniq_cut: float = 10.0,
                    stdev_min: float = 1e-8) -> tuple[list[str], pd.DataFrame]:
    """Drop near-zero-variance / quasi-constant features."""
    report = []
    kept = []
    for f in features:
        col = df[f].dropna()
        if len(col) == 0:
            report.append({"feature": f, "std": 0.0, "uniq_ratio": np.inf, "kept": False})
            continue
        std = float(col.std())
        # "NZV" = most-common-val count / second-most * uniqueness
        vc = col.value_counts()
        freq_ratio = vc.iloc[0] / vc.iloc[1] if len(vc) > 1 else np.inf
        if method == "stdev":
            passed = std > stdev_min
        else:
            passed = (freq_ratio < nzv_uniq_cut) and (std > stdev_min)
        report.append({"feature": f, "std": std, "freq_ratio": float(freq_ratio), "kept": bool(passed)})
        if passed:
            kept.append(f)
    return kept, pd.DataFrame(report)


# ===================================================================
# Stage B: pre-screening
# ===================================================================
def univariate_filter(
    X: np.ndarray, y: np.ndarray, feature_names: Sequence[str],
    method: str = "mwu_fdr", fdr_alpha: float = 0.20,
    multiclass_reduction: str = "min_p",
) -> tuple[list[str], pd.DataFrame]:
    """Mann-Whitney U + BH-FDR.

    For multiclass, run one-vs-rest per class and reduce p-values per feature
    using min_p (default).
    """
    classes = np.unique(y)
    p_matrix = np.ones((X.shape[1], max(1, len(classes))))
    for i, cls in enumerate(classes if len(classes) > 2 else [classes[-1]]):
        pos = y == cls
        neg = ~pos
        if pos.sum() < 2 or neg.sum() < 2:
            continue
        for j in range(X.shape[1]):
            vals = X[:, j]
            mask = ~np.isnan(vals)
            try:
                p = stats.mannwhitneyu(vals[mask & pos], vals[mask & neg], alternative="two-sided").pvalue
            except Exception:
                p = 1.0
            p_matrix[j, i] = p
    # Reduce
    if multiclass_reduction == "min_p":
        p_reduced = np.nanmin(p_matrix, axis=1)
    else:
        p_reduced = np.nanmean(p_matrix, axis=1)

    # BH-FDR
    order = np.argsort(p_reduced)
    m = len(p_reduced)
    thresh = fdr_alpha * (np.arange(1, m + 1) / m)
    sorted_p = p_reduced[order]
    passed_mask = np.zeros(m, dtype=bool)
    k = 0
    for idx in range(m):
        if sorted_p[idx] <= thresh[idx]:
            k = idx + 1
    if k > 0:
        pass_indices = order[:k]
        passed_mask[pass_indices] = True

    report = pd.DataFrame({
        "feature": list(feature_names),
        "pvalue": p_reduced,
        "kept": passed_mask,
    }).sort_values("pvalue").reset_index(drop=True)
    kept = report.loc[report["kept"], "feature"].tolist()
    return kept, report


# ===================================================================
# Stage C: redundancy
# ===================================================================
def correlation_filter(
    X: pd.DataFrame, features: Sequence[str],
    max_abs_corr: float = 0.90,
    group_by: Sequence[str] | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Greedy correlation pruning within groups (phase/roi).

    Within each group, compute |correlation|; for every pair above the
    threshold, drop the one with the larger mean |r| to other group members.
    """
    def _group_key(name: str) -> str:
        # feature names start with "{phase}_{roi}_..." so parse first two tokens
        parts = name.split("_", 2)
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        return "all"

    groups: dict[str, list[str]] = {}
    if group_by:
        for f in features:
            key = _group_key(f)
            groups.setdefault(key, []).append(f)
    else:
        groups["all"] = list(features)

    kept_all: list[str] = []
    report_rows = []
    for g, cols in groups.items():
        if len(cols) <= 1:
            kept_all.extend(cols)
            for c in cols:
                report_rows.append({"feature": c, "group": g, "dropped_due_to": None})
            continue
        sub = X[cols].apply(pd.to_numeric, errors="coerce")
        corr = sub.corr().abs().fillna(0.0)
        np.fill_diagonal(corr.values, 0.0)
        to_drop = set()
        # Mean absolute correlation to other features in group
        mean_abs = corr.mean(axis=1)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if corr.iloc[i, j] >= max_abs_corr:
                    # drop the one with larger overall correlation
                    drop = cols[i] if mean_abs.iloc[i] >= mean_abs.iloc[j] else cols[j]
                    to_drop.add(drop)
        for c in cols:
            if c in to_drop:
                report_rows.append({"feature": c, "group": g, "dropped_due_to": "high_corr"})
            else:
                kept_all.append(c)
                report_rows.append({"feature": c, "group": g, "dropped_due_to": None})

    return kept_all, pd.DataFrame(report_rows)


def mrmr_select(
    X: pd.DataFrame, y: np.ndarray, features: Sequence[str], k: int,
) -> tuple[list[str], pd.DataFrame]:
    """Maximum Relevance / Minimum Redundancy selection (F-statistic based).

    If the mrmr package is available, uses mrmr_classif. Otherwise falls
    back to a pure-numpy F-score + mean-abs-corr redundancy approximation.
    """
    k_eff = min(int(k), len(features))
    sub = X[list(features)].apply(pd.to_numeric, errors="coerce").fillna(X[list(features)].median())
    try:
        from mrmr import mrmr_classif
        selected = mrmr_classif(X=sub, y=pd.Series(y), K=k_eff, show_progress=False)
    except Exception:
        # Fallback: compute F-statistic relevance then greedily add features
        from sklearn.feature_selection import f_classif
        F, _ = f_classif(sub.values, y)
        order = np.argsort(F)[::-1]
        ranked = [features[i] for i in order]
        selected = [ranked[0]]
        corr = sub.corr().abs().fillna(0.0)
        while len(selected) < k_eff:
            best = None
            best_score = -np.inf
            for f in ranked:
                if f in selected:
                    continue
                relev = F[features.index(f)]
                redundancy = corr.loc[f, selected].mean() if selected else 0.0
                score = relev - redundancy
                if score > best_score:
                    best = f
                    best_score = score
            if best is None:
                break
            selected.append(best)

    report = pd.DataFrame({"feature": selected, "rank": range(1, len(selected) + 1)})
    return list(selected), report


# ===================================================================
# Stage D: stability selection
# ===================================================================
def stability_selection(
    X: np.ndarray, y: np.ndarray, feature_names: Sequence[str],
    method: str = "elastic_net",
    n_bootstrap: int = 50, sample_frac: float = 0.8,
    inclusion_threshold: float = 0.60,
    alpha_grid: Sequence[float] = (0.01, 0.1, 1.0),
    l1_ratio: Sequence[float] = (0.5,),
    min_features: int = 8, max_features: int = 40,
    random_state: int = 42,
) -> tuple[list[str], pd.DataFrame]:
    """Bootstrap resampling + sparse model selection -> inclusion frequency."""
    from sklearn.linear_model import ElasticNetCV, LassoCV, LogisticRegressionCV
    from sklearn.utils import check_random_state
    from sklearn.preprocessing import StandardScaler

    rng = check_random_state(random_state)
    freq = np.zeros(len(feature_names), dtype=float)
    n, p = X.shape
    n_sub = int(round(n * sample_frac))

    # Binarize y if more than 2 classes: use LabelBinarizer + multinomial?
    # Simpler: for multiclass, run stability selection per class (OVR) and OR-combine
    classes = np.unique(y)
    is_multiclass = len(classes) > 2

    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n_sub, replace=False)
        X_b, y_b = X[idx], y[idx]

        # Standardize within bootstrap
        sc = StandardScaler()
        X_bs = sc.fit_transform(np.nan_to_num(X_b))

        if is_multiclass:
            classes_here = np.unique(y_b)
            selected_any = np.zeros(p, dtype=bool)
            for cls in classes_here:
                y_bin = (y_b == cls).astype(int)
                if y_bin.sum() < 2 or (y_bin == 0).sum() < 2:
                    continue
                try:
                    if method == "lasso":
                        m = LassoCV(alphas=list(alpha_grid), cv=3, random_state=random_state).fit(X_bs, y_bin)
                        nonzero = np.abs(m.coef_) > 1e-8
                    elif method == "l1_logreg":
                        m = LogisticRegressionCV(Cs=[1.0 / a for a in alpha_grid],
                                                 cv=3, penalty="l1", solver="liblinear",
                                                 max_iter=1000, random_state=random_state).fit(X_bs, y_bin)
                        nonzero = np.abs(m.coef_[0]) > 1e-8
                    else:  # elastic_net
                        m = ElasticNetCV(alphas=list(alpha_grid), l1_ratio=list(l1_ratio),
                                         cv=3, random_state=random_state).fit(X_bs, y_bin)
                        nonzero = np.abs(m.coef_) > 1e-8
                    selected_any |= nonzero
                except Exception:
                    pass
            freq += selected_any
        else:
            y_bin = y_b.astype(int)
            try:
                if method == "lasso":
                    m = LassoCV(alphas=list(alpha_grid), cv=3, random_state=random_state).fit(X_bs, y_bin)
                    nonzero = np.abs(m.coef_) > 1e-8
                elif method == "l1_logreg":
                    m = LogisticRegressionCV(Cs=[1.0 / a for a in alpha_grid],
                                             cv=3, penalty="l1", solver="liblinear",
                                             max_iter=1000, random_state=random_state).fit(X_bs, y_bin)
                    nonzero = np.abs(m.coef_[0]) > 1e-8
                else:
                    m = ElasticNetCV(alphas=list(alpha_grid), l1_ratio=list(l1_ratio),
                                     cv=3, random_state=random_state).fit(X_bs, y_bin)
                    nonzero = np.abs(m.coef_) > 1e-8
                freq += nonzero
            except Exception:
                pass

    freq = freq / n_bootstrap
    report = pd.DataFrame({"feature": list(feature_names), "inclusion_freq": freq})
    report = report.sort_values("inclusion_freq", ascending=False).reset_index(drop=True)

    # Apply threshold with min/max clamping
    kept = report.loc[report["inclusion_freq"] >= inclusion_threshold, "feature"].tolist()
    if len(kept) < min_features:
        kept = report["feature"].head(min_features).tolist()
    if len(kept) > max_features:
        kept = report["feature"].head(max_features).tolist()
    report["kept"] = report["feature"].isin(set(kept))
    return kept, report
