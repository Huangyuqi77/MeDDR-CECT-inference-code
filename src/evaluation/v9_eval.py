"""Shared utilities for strict V9 HCC-vs-ICC experiments."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.evaluation.clinical import bootstrap_auc, calibration_metrics, decision_curve
from src.features import selection as FSEL

PHASES = ("P", "C1", "C2", "C3")
PAIRS = (("C2", "P"), ("C2", "C1"), ("C3", "C2"))
LONG_PAIRS = (("C1", "P"), ("C2", "C1"), ("C3", "C2"), ("C2", "P"), ("C3", "P"), ("C3", "C1"))
META_COLS = {"ID", "mask_source", "mask_transform_type", "mask_qc_flag", "roi", "radius_mm"}
LABEL_COLS = {"ID", "label_4clf", "label_stage1", "label_stage2", "label_name", "group", "center"}
EPS = 1e-3


def numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return non-label numeric columns that are safe to model."""
    return [
        c for c in df.columns
        if c not in LABEL_COLS
        and c not in META_COLS
        and c != "y_true"
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame()


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_labels(split_dir: Path) -> tuple[pd.DataFrame, set[str], set[str]]:
    labels = pd.read_csv(split_dir / "label_df_full.csv", encoding="utf-8-sig")
    labels["ID"] = labels["ID"].astype(str)
    train_ids = set(pd.read_csv(split_dir / "train_ids.csv", encoding="utf-8-sig")["ID"].astype(str))
    val_ids = set(pd.read_csv(split_dir / "val_ids.csv", encoding="utf-8-sig")["ID"].astype(str))
    return labels, train_ids, val_ids


def load_raw_features(cache_root: Path, phase: str, roi: str, cohort: str) -> pd.DataFrame:
    path = cache_root / f"features_{phase}_{roi}_{cohort}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["ID"] = df["ID"].astype(str)
    keep = ["ID"] + [c for c in df.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]
    return df[keep]


def load_prefixed_features(cache_root: Path, phase: str, roi: str, cohort: str) -> pd.DataFrame:
    df = load_raw_features(cache_root, phase, roi, cohort)
    return df.rename(columns={c: f"{phase}_{roi}_{c}" for c in df.columns if c != "ID"})


def build_dynamic_features(cache_root: Path, cohort: str, stems: Sequence[str], roi: str = "tumor",
                           pairs: Sequence[tuple[str, str]] = PAIRS) -> pd.DataFrame:
    phase = {p: load_raw_features(cache_root, p, roi, cohort).set_index("ID") for p in PHASES}
    ids = sorted(set.intersection(*[set(x.index) for x in phase.values()]))
    out = pd.DataFrame({"ID": ids})
    cols = {}
    for stem in stems:
        for a, b in pairs:
            if stem not in phase[a].columns or stem not in phase[b].columns:
                continue
            va = pd.to_numeric(phase[a][stem].reindex(ids), errors="coerce").to_numpy(dtype=float)
            vb = pd.to_numeric(phase[b][stem].reindex(ids), errors="coerce").to_numpy(dtype=float)
            cols[f"DYN_delta_{a}_{b}_{stem}"] = va - vb
            denom = np.where(np.abs(vb) > EPS, np.abs(vb), EPS)
            cols[f"DYN_ratio_{a}_{b}_{stem}"] = (va - vb) / denom
    if cols:
        out = pd.concat([out, pd.DataFrame(cols)], axis=1)
    return out


def stage2_frames(labels: pd.DataFrame, train_ids: set[str], val_ids: set[str],
                  int_feat: pd.DataFrame, ext_feat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lab = labels[["ID", "label_4clf", "label_stage1", "label_name", "center"]].copy()
    internal = lab[lab["center"] == "internal"].merge(int_feat, on="ID", how="inner")
    external = lab[lab["center"] == "external"].merge(ext_feat, on="ID", how="inner")
    internal = internal[internal["label_stage1"] == 1].copy()
    external = external[external["label_4clf"].isin([0, 1])].copy()
    train = internal[internal["ID"].isin(train_ids)].copy()
    val = internal[internal["ID"].isin(val_ids)].copy()
    return train, val, external


def y_arrays(train: pd.DataFrame, val: pd.DataFrame, ext: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        train["label_4clf"].astype(int).to_numpy(),
        val["label_4clf"].astype(int).to_numpy(),
        ext["label_4clf"].astype(int).to_numpy(),
    )


def hcc_icc_score_from_proba(proba: np.ndarray, classes: Sequence[int]) -> np.ndarray:
    classes = list(classes)
    if 0 in classes and 1 in classes:
        p = proba[:, [classes.index(0), classes.index(1)]]
        p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
        return p[:, 1]
    return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]


def auc_hcc_icc(y: Sequence[int], score: Sequence[float]) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(score, dtype=float)
    mask = np.isin(y, [0, 1]) & np.isfinite(s)
    if mask.sum() < 3 or len(np.unique(y[mask])) < 2:
        return float("nan")
    return float(roc_auc_score(y[mask], s[mask]))


def lock_threshold(y_val: Sequence[int], score_val: Sequence[float]) -> float:
    y = np.asarray(y_val).astype(int)
    s = np.asarray(score_val, dtype=float)
    mask = np.isin(y, [0, 1]) & np.isfinite(s)
    fpr, tpr, thr = roc_curve(y[mask], s[mask])
    idx = int(np.argmax(tpr - fpr))
    out = float(thr[idx])
    return 0.5 if not np.isfinite(out) else out


def clinical_ops(y_true: Sequence[int], score: Sequence[float], threshold: float) -> dict:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    mask = np.isin(y, [0, 1]) & np.isfinite(s)
    y = y[mask]
    s = s[mask]
    pred = (s >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    ppv = tp / max(1, tp + fp)
    npv = tn / max(1, tn + fn)
    return {
        "n_evaluable": int(len(y)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
        "accuracy": float((tp + tn) / max(1, tp + tn + fp + fn)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def metric_row(method: str, mode: str, y_val: Sequence[int], score_val: Sequence[float],
               y_ext: Sequence[int], score_ext: Sequence[float], n_boot: int, seed: int,
               extra: dict | None = None) -> dict:
    threshold = lock_threshold(y_val, score_val)
    y = np.asarray(y_ext).astype(int)
    s = np.asarray(score_ext, dtype=float)
    mask = np.isin(y, [0, 1]) & np.isfinite(s)
    auc = bootstrap_auc(y[mask], s[mask], n_boot=n_boot, seed=seed)
    row = {"mode": mode, "method": method, "threshold_locked": threshold, **auc, **clinical_ops(y, s, threshold)}
    if extra:
        row.update(extra)
    return row


def calibration_row(method: str, mode: str, y_ext: Sequence[int], score_ext: Sequence[float]) -> tuple[dict, pd.DataFrame]:
    y = np.asarray(y_ext).astype(int)
    s = np.asarray(score_ext, dtype=float)
    mask = np.isin(y, [0, 1]) & np.isfinite(s)
    metrics, curve = calibration_metrics(y[mask], s[mask])
    metrics.update({"mode": mode, "method": method})
    curve.insert(0, "method", method)
    curve.insert(0, "mode", mode)
    return metrics, curve


def dca_rows(method: str, mode: str, y_ext: Sequence[int], score_ext: Sequence[float]) -> pd.DataFrame:
    y = np.asarray(y_ext).astype(int)
    s = np.asarray(score_ext, dtype=float)
    mask = np.isin(y, [0, 1]) & np.isfinite(s)
    df = decision_curve(y[mask], s[mask])
    df.insert(0, "method", method)
    df.insert(0, "mode", mode)
    return df


def prep_numeric(train: pd.DataFrame, val: pd.DataFrame, ext: pd.DataFrame, features: Sequence[str]):
    features = list(features)
    xtr = train[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    xva = val[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    xex = ext[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    imp = SimpleImputer(strategy="median").fit(xtr)
    xtr = imp.transform(xtr)
    xva = imp.transform(xva)
    xex = imp.transform(xex)
    sc = StandardScaler().fit(xtr)
    return sc.transform(xtr), sc.transform(xva), sc.transform(xex), imp, sc


def select_train_features(train: pd.DataFrame, y_train: Sequence[int], pool: Sequence[str],
                          k: int = 12, fdr_alpha: float = 0.20,
                          corr: float = 0.90) -> tuple[list[str], pd.DataFrame]:
    pool = [c for c in pool if c in train.columns]
    reports = []
    if not pool:
        return [], pd.DataFrame()
    kept, r = FSEL.missing_filter(train, pool, max_rate=0.30)
    r["stage"] = "missing"
    reports.append(r)
    kept, r = FSEL.variance_filter(train, kept, method="stdev", stdev_min=1e-8)
    r["stage"] = "variance"
    reports.append(r)
    if not kept:
        return [], pd.concat(reports, ignore_index=True, sort=False)
    x = train[kept].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    uni, r = FSEL.univariate_filter(x, np.asarray(y_train).astype(int), kept, fdr_alpha=fdr_alpha, multiclass_reduction="min_p")
    r["stage"] = "univariate"
    reports.append(r)
    if len(uni) >= min(k, 3):
        kept = uni
    kept, r = FSEL.correlation_filter(train, kept, max_abs_corr=corr)
    r["stage"] = "correlation"
    reports.append(r)
    if not kept:
        return [], pd.concat(reports, ignore_index=True, sort=False)
    selected, r = FSEL.mrmr_select(train, np.asarray(y_train).astype(int), kept, k=min(k, len(kept)))
    r["stage"] = "mrmr"
    reports.append(r)
    return selected, pd.concat(reports, ignore_index=True, sort=False)


def fit_sklearn_model(train: pd.DataFrame, val: pd.DataFrame, ext: pd.DataFrame,
                      features: Sequence[str], model_kind: str, seed: int) -> dict:
    y_train, y_val, y_ext = y_arrays(train, val, ext)
    xtr, xva, xex, imp, sc = prep_numeric(train, val, ext, features)
    candidates = []
    if model_kind == "LR_L2":
        for c in (0.01, 0.1, 1.0, 10.0):
            model = LogisticRegression(penalty="l2", C=c, solver="lbfgs", class_weight="balanced",
                                       max_iter=3000, random_state=seed)
            candidates.append((f"LR_L2_C{c}", model))
    elif model_kind == "ElasticNet":
        for c in (0.01, 0.1, 1.0, 10.0):
            for l1 in (0.2, 0.5, 0.8):
                model = LogisticRegression(penalty="elasticnet", C=c, l1_ratio=l1, solver="saga",
                                           class_weight="balanced", max_iter=5000, random_state=seed)
                candidates.append((f"ElasticNet_C{c}_l1{l1}", model))
    elif model_kind == "MLP":
        for h in (8, 16):
            for alpha in (1e-4, 1e-3, 1e-2):
                model = MLPClassifier(hidden_layer_sizes=(h,), alpha=alpha, activation="relu",
                                      max_iter=800, early_stopping=True, n_iter_no_change=30,
                                      random_state=seed)
                candidates.append((f"MLP_h{h}_a{alpha}", model))
    else:
        raise ValueError(model_kind)

    best = None
    for tag, model in candidates:
        model.fit(xtr, y_train)
        val_score = hcc_icc_score_from_proba(model.predict_proba(xva), model.classes_)
        val_auc = auc_hcc_icc(y_val, val_score)
        if best is None or val_auc > best["val_auc"]:
            best = {"tag": tag, "model": model, "val_auc": val_auc, "val_score": val_score}
    assert best is not None
    ext_score = hcc_icc_score_from_proba(best["model"].predict_proba(xex), best["model"].classes_)
    return {
        "model": best["model"],
        "model_tag": best["tag"],
        "model_kind": model_kind,
        "features": list(features),
        "val_auc": float(best["val_auc"]),
        "val_score": best["val_score"],
        "ext_score": ext_score,
        "imputer": imp,
        "scaler": sc,
    }


def paired_bootstrap_delta(y: Sequence[int], score_a: Sequence[float], score_b: Sequence[float],
                           n_boot: int = 2000, seed: int = 42) -> dict:
    y = np.asarray(y).astype(int)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    mask = np.isin(y, [0, 1]) & np.isfinite(a) & np.isfinite(b)
    y, a, b = y[mask], a[mask], b[mask]
    delta = float(roc_auc_score(y, a) - roc_auc_score(y, b))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(float(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx])))
    lo, hi = np.quantile(vals, [0.025, 0.975]) if vals else (np.nan, np.nan)
    p = float(2 * min(np.mean(np.asarray(vals) <= 0), np.mean(np.asarray(vals) >= 0))) if vals else np.nan
    return {"n": int(len(y)), "delta_auc": delta, "delta_ci_low": float(lo), "delta_ci_high": float(hi), "p_two_sided": p}


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / max(1, len(a) + len(b) - 2))
    return float((np.mean(b) - np.mean(a)) / pooled) if pooled > 0 else np.nan


def direction_consistency(train: pd.DataFrame, ext: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    rows = []
    for feat in features:
        if feat not in train.columns or feat not in ext.columns:
            continue
        tr0 = train.loc[train["label_4clf"].astype(int) == 0, feat].astype(float).to_numpy()
        tr1 = train.loc[train["label_4clf"].astype(int) == 1, feat].astype(float).to_numpy()
        ex0 = ext.loc[ext["label_4clf"].astype(int) == 0, feat].astype(float).to_numpy()
        ex1 = ext.loc[ext["label_4clf"].astype(int) == 1, feat].astype(float).to_numpy()
        d_int = cohen_d(tr0, tr1)
        d_ext = cohen_d(ex0, ex1)
        rows.append({
            "mode": "external_final_only",
            "feature": feat,
            "internal_HCC_mean": float(np.nanmean(tr0)) if len(tr0) else np.nan,
            "internal_ICC_mean": float(np.nanmean(tr1)) if len(tr1) else np.nan,
            "external_HCC_mean": float(np.nanmean(ex0)) if len(ex0) else np.nan,
            "external_ICC_mean": float(np.nanmean(ex1)) if len(ex1) else np.nan,
            "internal_cohen_d": d_int,
            "external_cohen_d": d_ext,
            "sign_match": bool(np.sign(d_int) == np.sign(d_ext)) if np.isfinite(d_int) and np.isfinite(d_ext) else False,
            "direction_reversal_flag": bool(np.sign(d_int) != np.sign(d_ext)) if np.isfinite(d_int) and np.isfinite(d_ext) else True,
        })
    return pd.DataFrame(rows)


def score_correlations(original: np.ndarray, perturbed: np.ndarray) -> dict:
    mask = np.isfinite(original) & np.isfinite(perturbed)
    if mask.sum() < 3:
        return {"spearman": np.nan, "pearson": np.nan}
    return {
        "spearman": float(spearmanr(original[mask], perturbed[mask]).statistic),
        "pearson": float(pearsonr(original[mask], perturbed[mask]).statistic),
    }


def align_by_id(left: pd.DataFrame, right: pd.DataFrame, how: str = "inner") -> pd.DataFrame:
    """Merge two feature blocks on ID without duplicating non-ID columns."""
    if left.empty:
        return right.copy()
    if right.empty:
        return left.copy()
    overlap = sorted((set(left.columns) & set(right.columns)) - {"ID"})
    if overlap:
        right = right.drop(columns=overlap)
    return left.merge(right, on="ID", how=how)


def merge_feature_blocks(blocks: Sequence[pd.DataFrame], how: str = "outer") -> pd.DataFrame:
    out = pd.DataFrame()
    for block in blocks:
        if block is None or block.empty:
            continue
        b = block.copy()
        b["ID"] = b["ID"].astype(str).str.strip()
        out = align_by_id(out, b, how=how)
    return out


def save_scores(path: Path, ids: Sequence[str], y_true: Sequence[int], scores: dict[str, Sequence[float]]) -> pd.DataFrame:
    df = pd.DataFrame({"ID": list(ids), "y_true": np.asarray(y_true).astype(int)})
    for method, score in scores.items():
        df[method] = np.asarray(score, dtype=float)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def row_from_existing_scores(method: str, y_ext: Sequence[int], score_ext: Sequence[float],
                             y_val: Sequence[int], score_val: Sequence[float],
                             n_boot: int, seed: int, extra: dict | None = None) -> dict:
    return metric_row(
        method=method,
        mode="strict_inductive",
        y_val=y_val,
        score_val=score_val,
        y_ext=y_ext,
        score_ext=score_ext,
        n_boot=n_boot,
        seed=seed,
        extra=extra,
    )


def summarize_feature_families(features: Sequence[str]) -> str:
    vals = []
    for f in features:
        name = str(f)
        if name.startswith("DYN_") or "_delta_" in name or "_ratio_" in name:
            vals.append("delta_ratio")
        elif "surrogate" in name or name.startswith(("tumor_mean_", "tumor_median_", "tumor_p90_", "tumor_liver_", "rim_core_")):
            vals.append("enhancement_surrogate")
        elif "_peri" in name:
            vals.append("peritumoral")
        elif "Imc2" in name:
            vals.append("imc2_anchor")
        else:
            vals.append("radiomics")
    return "|".join(sorted(set(vals)))


def read_v8_manifest(output_root: Path) -> dict:
    path = output_root / "v8_strict" / "v8_main" / "v8_manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def common_score_frame(val_ids: Sequence[str], y_val: Sequence[int], ext_ids: Sequence[str], y_ext: Sequence[int],
                       method_to_val_ext: dict[str, tuple[Sequence[float], Sequence[float]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    val = pd.DataFrame({"ID": list(val_ids), "y_true": np.asarray(y_val).astype(int)})
    ext = pd.DataFrame({"ID": list(ext_ids), "y_true": np.asarray(y_ext).astype(int)})
    for method, (sv, se) in method_to_val_ext.items():
        val[method] = np.asarray(sv, dtype=float)
        ext[method] = np.asarray(se, dtype=float)
    return val, ext


def model_complexity(model) -> int:
    """Best-effort parameter count for sklearn and torch models."""
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            return int(sum(p.numel() for p in model.parameters()))
    except Exception:
        pass
    total = 0
    for attr in ("coef_", "intercept_"):
        val = getattr(model, attr, None)
        if val is not None:
            total += int(np.asarray(val).size)
    if hasattr(model, "coefs_"):
        total += sum(int(np.asarray(w).size) for w in model.coefs_)
    if hasattr(model, "intercepts_"):
        total += sum(int(np.asarray(b).size) for b in model.intercepts_)
    return int(total)
