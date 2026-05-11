"""Shared helpers for V9 mainline validation audits.

The helpers in this module replay V9 surrogate candidates from internal
train/validation data and then score external original/perturbed feature
tables without refitting on external data.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.evaluation.clinical import bootstrap_auc, calibration_metrics, decision_curve
from src.evaluation.score_fusion import fit_score_fusions
from src.evaluation.v9_eval import (
    EPS,
    PHASES,
    auc_hcc_icc,
    build_dynamic_features,
    calibration_row,
    clinical_ops,
    direction_consistency,
    dca_rows,
    fit_sklearn_model,
    hcc_icc_score_from_proba,
    load_labels,
    load_prefixed_features,
    load_raw_features,
    lock_threshold,
    merge_feature_blocks,
    metric_row,
    numeric_feature_cols,
    paired_bootstrap_delta,
    read_v8_manifest,
    select_train_features,
    stage2_frames,
    summarize_feature_families,
    y_arrays,
)
from src.features.enhancement_surrogates import STAT_STEMS
from src.utils.config import resolve_path


VARIANTS = ("original", "dilate_1px", "erode_1px")
PERTURB_MASK_ROOT = Path("D:/Onekey/cache/plc_perturb_masks")
V9_ROOT_NAME = "v9_dynamic_upgrade"
VALIDATION_ROOT_NAME = "v9_mainline_validation"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame()


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _ids_from_tables(tables: dict[str, pd.DataFrame]) -> list[str]:
    ids: set[str] = set()
    for df in tables.values():
        if not df.empty and "ID" in df.columns:
            ids.update(df["ID"].astype(str).str.strip().tolist())
    return sorted(ids)


def _phase_value(tables: dict[str, pd.DataFrame], phase: str, stem: str, ids: list[str]) -> np.ndarray:
    df = tables.get(phase, pd.DataFrame())
    if df.empty or stem not in df.columns:
        return np.full(len(ids), np.nan, dtype=float)
    s = df.set_index("ID")[stem].reindex(ids)
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return num / np.where(np.abs(den) > EPS, np.abs(den), EPS)


def build_surrogate_from_tables(tumor: dict[str, pd.DataFrame], liver: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ids = _ids_from_tables({**{f"tumor_{k}": v for k, v in tumor.items()}, **{f"liver_{k}": v for k, v in liver.items()}})
    out = pd.DataFrame({"ID": ids})
    tvals: dict[tuple[str, str], np.ndarray] = {}
    lmean: dict[str, np.ndarray] = {}
    for ph in PHASES:
        for stat, stem in STAT_STEMS.items():
            tvals[(ph, stat)] = _phase_value(tumor, ph, stem, ids)
        lmean[ph] = _phase_value(liver, ph, STAT_STEMS["mean"], ids)

    for prefix, stat, pairs in [
        ("tumor_mean", "mean", [("C1", "P"), ("C2", "C1"), ("C3", "C2"), ("C2", "P"), ("C3", "P")]),
        ("tumor_median", "median", [("C1", "P"), ("C2", "C1"), ("C3", "C2")]),
        ("tumor_p90", "p90", [("C1", "P"), ("C2", "C1"), ("C3", "C2")]),
    ]:
        for a, b in pairs:
            out[f"{prefix}_{a}_minus_{b}"] = tvals[(a, stat)] - tvals[(b, stat)]

    for a, b in [("C1", "P"), ("C2", "C1"), ("C3", "C2"), ("C2", "P"), ("C3", "P")]:
        out[f"tumor_ratio_{a}_{b}"] = _safe_ratio(tvals[(a, "mean")] - tvals[(b, "mean")], tvals[(b, "mean")])

    contrast = {}
    for ph in PHASES:
        out[f"tumor_liver_mean_ratio_{ph}"] = _safe_ratio(tvals[(ph, "mean")], lmean[ph])
        contrast[ph] = tvals[(ph, "mean")] - lmean[ph]
    for a, b in [("C1", "P"), ("C2", "C1"), ("C3", "C2"), ("C2", "P"), ("C3", "P")]:
        out[f"tumor_liver_delta_{a}_{b}"] = contrast[a] - contrast[b]
    return out


def variant_cache_root(cache_root: Path, variant: str) -> Path:
    return cache_root if variant == "original" else resolve_path("cache/radiomics_perturb") / variant


def load_tumor_tables(cache_root: Path, cohort: str, variant: str = "original") -> dict[str, pd.DataFrame]:
    root = variant_cache_root(cache_root, variant)
    out = {}
    for ph in PHASES:
        path = root / f"features_{ph}_tumor_{cohort}.csv"
        if path.exists():
            df = pd.read_csv(path, encoding="utf-8-sig")
            df["ID"] = df["ID"].astype(str)
            out[ph] = df
        else:
            out[ph] = pd.DataFrame()
    return out


def load_liver_tables(cache_root: Path, cohort: str) -> dict[str, pd.DataFrame]:
    out = {}
    for ph in PHASES:
        try:
            out[ph] = load_raw_features(cache_root, ph, "liver", cohort)
        except FileNotFoundError:
            out[ph] = pd.DataFrame()
    return out


def build_external_surrogate_variant(cfg: dict, variant: str, out_dir: Path, overwrite: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build or load external V9 surrogate features for one perturbation variant."""
    output_root = resolve_path(cfg["output_root"])
    cache_root = resolve_path(cfg["feature_cache_root"])
    if variant == "original":
        src = output_root / V9_ROOT_NAME / "enhancement_surrogates" / "surrogate_features_external.csv"
        core_rep = output_root / V9_ROOT_NAME / "enhancement_surrogates" / "core_rim_case_report_external.csv"
        return pd.read_csv(src, encoding="utf-8-sig"), read_csv(core_rep)

    feat_path = out_dir / "features" / f"v9_surrogate_features_external_{variant}.csv"
    rep_path = out_dir / "features" / f"v9_core_rim_case_report_external_{variant}.csv"
    if feat_path.exists() and rep_path.exists() and not overwrite:
        return pd.read_csv(feat_path, encoding="utf-8-sig"), pd.read_csv(rep_path, encoding="utf-8-sig")

    tumor = load_tumor_tables(cache_root, "external", variant)
    liver = load_liver_tables(cache_root, "external")
    base = build_surrogate_from_tables(tumor, liver)
    ids = sorted(base["ID"].astype(str).str.strip().unique())
    data_root = resolve_path(cfg["external_root"])
    mask_root = PERTURB_MASK_ROOT / variant / "external"
    core, report = compute_core_rim_stats_with_mask_root(
        ids,
        cohort="external",
        data_root=data_root,
        mask_root=mask_root,
        cache_path=out_dir / "features" / f"v9_core_rim_stats_external_{variant}.csv",
        report_path=rep_path,
        overwrite=overwrite,
    )
    merged = base.merge(core, on="ID", how="left")
    feat_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(feat_path, index=False, encoding="utf-8-sig")
    report.to_csv(rep_path, index=False, encoding="utf-8-sig")
    return merged, report


def compute_core_rim_stats_with_mask_root(
    ids: Sequence[str],
    cohort: str,
    data_root: Path,
    mask_root: Path,
    cache_path: Path,
    report_path: Path,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute core/rim means using an explicit perturbation mask root."""
    import SimpleITK as sitk

    if cache_path.exists() and report_path.exists() and not overwrite:
        return pd.read_csv(cache_path, encoding="utf-8-sig"), pd.read_csv(report_path, encoding="utf-8-sig")

    rows: dict[str, dict] = {}
    reports = []
    for patient_id in [str(x).strip() for x in ids]:
        rec = {"ID": patient_id}
        for ph in PHASES:
            img_p = data_root / ph / "images" / f"{patient_id}.nii.gz"
            msk_p = mask_root / ph / "masks" / f"{patient_id}.nii.gz"
            try:
                if not img_p.exists():
                    raise FileNotFoundError(f"image_missing:{img_p}")
                if not msk_p.exists():
                    raise FileNotFoundError(f"perturbed_mask_missing:{msk_p}")
                img = sitk.ReadImage(str(img_p))
                msk = sitk.ReadImage(str(msk_p))
                if img.GetSize() != msk.GetSize():
                    msk = sitk.Resample(msk, img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, msk.GetPixelID())
                arr = sitk.GetArrayFromImage(img).astype(float)
                marr = sitk.GetArrayFromImage(msk) > 0
                tumor_voxels = int(marr.sum())
                if tumor_voxels <= 0:
                    raise ValueError("empty_tumor_mask")
                msk_bin = sitk.Cast(msk > 0, sitk.sitkUInt8)
                core_img = sitk.BinaryErode(msk_bin, [1, 1, 1], 1)
                core = sitk.GetArrayFromImage(core_img) > 0
                rim = marr & ~core
                core_voxels = int(core.sum())
                rim_voxels = int(rim.sum())
                if core_voxels <= 0 or rim_voxels <= 0:
                    raise ValueError(f"empty_core_or_rim:core={core_voxels};rim={rim_voxels}")
                rec[f"rim_core_mean_delta_{ph}"] = float(np.nanmean(arr[rim]) - np.nanmean(arr[core]))
                reports.append({
                    "ID": patient_id,
                    "cohort": cohort,
                    "phase": ph,
                    "status": "ok",
                    "image_path": str(img_p),
                    "mask_path": str(msk_p),
                    "tumor_voxels": tumor_voxels,
                    "core_voxels": core_voxels,
                    "rim_voxels": rim_voxels,
                    "error": "",
                })
            except Exception as exc:
                rec[f"rim_core_mean_delta_{ph}"] = np.nan
                reports.append({
                    "ID": patient_id,
                    "cohort": cohort,
                    "phase": ph,
                    "status": "failed",
                    "image_path": str(img_p),
                    "mask_path": str(msk_p),
                    "tumor_voxels": np.nan,
                    "core_voxels": np.nan,
                    "rim_voxels": np.nan,
                    "error": str(exc),
                })
        rows[patient_id] = rec
        if len(reports) % 50 == 0:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows.values()).to_csv(cache_path, index=False, encoding="utf-8-sig")
            pd.DataFrame(reports).to_csv(report_path, index=False, encoding="utf-8-sig")

    out = pd.DataFrame(rows.values())
    for a, b in [("C1", "P"), ("C2", "C1"), ("C3", "C2"), ("C3", "C1")]:
        out[f"rim_core_dynamic_{a}_minus_{b}"] = pd.to_numeric(out.get(f"rim_core_mean_delta_{a}"), errors="coerce") - pd.to_numeric(out.get(f"rim_core_mean_delta_{b}"), errors="coerce")
    rep = pd.DataFrame(reports)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False, encoding="utf-8-sig")
    rep.to_csv(report_path, index=False, encoding="utf-8-sig")
    return out, rep


def build_original_blocks(cfg: dict) -> dict:
    output_root = resolve_path(cfg["output_root"])
    cache_root = resolve_path(cfg["feature_cache_root"])
    split_dir = resolve_path(cfg["split_root"])
    labels, train_ids, val_ids = load_labels(split_dir)
    v8 = read_v8_manifest(output_root)
    stems = list(v8["v8_stems_top_k"])
    sur_i = pd.read_csv(output_root / V9_ROOT_NAME / "enhancement_surrogates" / "surrogate_features_internal.csv", encoding="utf-8-sig")
    sur_e = pd.read_csv(output_root / V9_ROOT_NAME / "enhancement_surrogates" / "surrogate_features_external.csv", encoding="utf-8-sig")
    c2_i = load_prefixed_features(cache_root, "C2", "tumor", "internal")
    c2_e = load_prefixed_features(cache_root, "C2", "tumor", "external")
    dyn_i = build_dynamic_features(cache_root, "internal", stems)
    dyn_e = build_dynamic_features(cache_root, "external", stems)
    int_feat = merge_feature_blocks([sur_i, c2_i, dyn_i])
    ext_feat = merge_feature_blocks([sur_e, c2_e, dyn_e])
    train, val, ext = stage2_frames(labels, train_ids, val_ids, int_feat, ext_feat)
    return {
        "labels": labels,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "v8_manifest": v8,
        "train": train,
        "val": val,
        "ext": ext,
        "surrogate_internal": sur_i,
        "surrogate_external": sur_e,
        "dynamic_internal": dyn_i,
        "dynamic_external": dyn_e,
        "c2_internal": c2_i,
        "c2_external": c2_e,
    }


def build_external_feature_frame_for_variant(cfg: dict, variant: str, surrogate: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    output_root = resolve_path(cfg["output_root"])
    cache_root = resolve_path(cfg["feature_cache_root"])
    v8 = read_v8_manifest(output_root)
    stems = list(v8["v8_stems_top_k"])
    root = variant_cache_root(cache_root, variant)
    dyn = build_dynamic_features(root, "external", stems)
    c2 = load_prefixed_features(root, "C2", "tumor", "external")
    feat = merge_feature_blocks([surrogate, c2, dyn])
    lab = labels[["ID", "label_4clf", "label_stage1", "label_name", "center"]].copy()
    ext = lab[lab["center"] == "external"].merge(feat, on="ID", how="inner")
    ext = ext[ext["label_4clf"].isin([0, 1])].copy()
    return ext


def score_artifact(artifact: dict, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    features = artifact["features"]
    x = frame[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    missing_before = np.isnan(x).sum(axis=1)
    x = artifact["imputer"].transform(x)
    x = artifact["scaler"].transform(x)
    score = hcc_icc_score_from_proba(artifact["model"].predict_proba(x), artifact["model"].classes_)
    return np.asarray(score, dtype=float), np.asarray(missing_before, dtype=int)


def fit_locked_candidates(cfg: dict, out_dir: Path, seed: int = 42, overwrite: bool = False) -> dict:
    model_dir = ensure_dir(resolve_path(cfg["output_root"]) / VALIDATION_ROOT_NAME / "models")
    artifact_path = model_dir / "v9_locked_candidate_artifacts.pkl"
    if artifact_path.exists() and not overwrite:
        with artifact_path.open("rb") as f:
            return pickle.load(f)

    blocks = build_original_blocks(cfg)
    train, val, ext = blocks["train"], blocks["val"], blocks["ext"]
    y_train, _, _ = y_arrays(train, val, ext)
    val_mask = val["label_4clf"].isin([0, 1]).to_numpy()
    y_val = val.loc[val_mask, "label_4clf"].astype(int).to_numpy()
    y_ext = ext["label_4clf"].astype(int).to_numpy()
    v9_root = resolve_path(cfg["output_root"]) / V9_ROOT_NAME
    sur_table = read_csv(v9_root / "enhancement_surrogates" / "surrogate_model_manifest_table.csv")
    phase_metrics = read_csv(v9_root / "phase_token" / "phase_token_metrics.csv")

    candidate_rows = []
    for method in [
        "S0_surrogate_only_ElasticNet",
        "S2_V8_plus_surrogate_concat_LR_L2",
        "S3_V8_plus_surrogate_score_fusion",
    ]:
        row = sur_table[sur_table["method"].astype(str) == method]
        if not row.empty:
            candidate_rows.append(row.iloc[0].to_dict())
    row = phase_metrics[phase_metrics["method"].astype(str) == "V9-B-surrogate+V8"]
    if not row.empty:
        rec = row.iloc[0].to_dict()
        rec["method"] = "V9-B-surrogate+V8"
        rec["model_kind"] = "LR_L2"
        candidate_rows.append(rec)

    imc2 = "C2_tumor_logarithm_glcm_Imc2"
    ref_defs = [
        {"method": "Imc2_only", "model_kind": "LR_L2", "features": imc2},
        {"method": "V8_dyn_only", "model_kind": "LR_L2", "features": "|".join(blocks["v8_manifest"]["v8_dynamic_selected"])},
    ]
    candidate_rows = ref_defs + candidate_rows

    artifacts: dict[str, dict] = {}
    val_scores = pd.DataFrame({"ID": val.loc[val_mask, "ID"].astype(str).tolist(), "y_true": y_val})
    ext_scores = pd.DataFrame({"ID": ext["ID"].astype(str).tolist(), "y_true": y_ext})
    fit_rows = []
    for rec in candidate_rows:
        method = str(rec["method"])
        if method == "S3_V8_plus_surrogate_score_fusion":
            continue
        features = [f for f in str(rec.get("features", "")).split("|") if f and f in train.columns and f in ext.columns]
        if not features:
            continue
        kind = str(rec.get("model_kind", "LR_L2"))
        if kind not in {"LR_L2", "ElasticNet", "MLP"}:
            kind = "LR_L2"
        fit = fit_sklearn_model(train, val, ext, features, kind, seed)
        threshold = lock_threshold(y_val, fit["val_score"][val_mask])
        fit.update({
            "method": method,
            "threshold": threshold,
            "y_val": y_val,
            "y_ext": y_ext,
            "val_ids": val.loc[val_mask, "ID"].astype(str).tolist(),
            "ext_ids": ext["ID"].astype(str).tolist(),
            "feature_families": summarize_feature_families(features),
        })
        artifacts[method] = fit
        val_scores[method] = fit["val_score"][val_mask]
        ext_scores[method] = fit["ext_score"]
        fit_rows.append({
            "method": method,
            "model_kind": fit["model_kind"],
            "model_tag": fit["model_tag"],
            "n_features": len(features),
            "features": "|".join(features),
            "val_auc": fit["val_auc"],
            "threshold_locked": threshold,
        })

    if "V8_dyn_only" in artifacts and "S0_surrogate_only_ElasticNet" in artifacts:
        v8_val = val_scores["V8_dyn_only"].to_numpy(dtype=float)
        sur_val = val_scores["S0_surrogate_only_ElasticNet"].to_numpy(dtype=float)
        v8_ext = ext_scores["V8_dyn_only"].to_numpy(dtype=float)
        sur_ext = ext_scores["S0_surrogate_only_ElasticNet"].to_numpy(dtype=float)
        v8_gain = max(auc_hcc_icc(y_val, v8_val) - 0.5, 1e-6)
        sur_gain = max(auc_hcc_icc(y_val, sur_val) - 0.5, 1e-6)
        w_sur = sur_gain / (v8_gain + sur_gain)
        w_v8 = 1.0 - w_sur
        score_val = w_v8 * v8_val + w_sur * sur_val
        score_ext = w_v8 * v8_ext + w_sur * sur_ext
        artifacts["S3_V8_plus_surrogate_score_fusion"] = {
            "method": "S3_V8_plus_surrogate_score_fusion",
            "type": "score_fusion",
            "branches": ["V8_dyn_only", "S0_surrogate_only_ElasticNet"],
            "weights": {"V8_dyn_only": w_v8, "S0_surrogate_only_ElasticNet": w_sur},
            "threshold": lock_threshold(y_val, score_val),
            "val_score": score_val,
            "ext_score": score_ext,
            "y_val": y_val,
            "y_ext": y_ext,
            "val_ids": val_scores["ID"].astype(str).tolist(),
            "ext_ids": ext_scores["ID"].astype(str).tolist(),
            "features": artifacts["V8_dyn_only"]["features"] + artifacts["S0_surrogate_only_ElasticNet"]["features"],
            "feature_families": "delta_ratio|enhancement_surrogate",
        }
        val_scores["S3_V8_plus_surrogate_score_fusion"] = score_val
        ext_scores["S3_V8_plus_surrogate_score_fusion"] = score_ext
        fit_rows.append({
            "method": "S3_V8_plus_surrogate_score_fusion",
            "model_kind": "score_fusion",
            "model_tag": "val_auc_weighted_two_branch",
            "n_features": len(artifacts["S3_V8_plus_surrogate_score_fusion"]["features"]),
            "features": "|".join(artifacts["S3_V8_plus_surrogate_score_fusion"]["features"]),
            "threshold_locked": artifacts["S3_V8_plus_surrogate_score_fusion"]["threshold"],
            "fusion_weight_v8": w_v8,
            "fusion_weight_surrogate": w_sur,
        })

    # Recompute score fusions that can be replayed without phase-token torch artifacts.
    fusion_methods = ["Imc2_only", "V8_dyn_only", "S0_surrogate_only_ElasticNet", "V9-B-surrogate+V8"]
    avail = [m for m in fusion_methods if m in val_scores.columns and m in ext_scores.columns]
    if len(avail) >= 2:
        fusions = fit_score_fusions(val_scores[["ID", "y_true"] + avail], ext_scores[["ID", "y_true"] + avail], avail, seed=seed)
        for name, rec in fusions.items():
            artifacts[f"Best_V9_score_fusion::{name}"] = {
                "method": f"Best_V9_score_fusion::{name}",
                "type": "score_fusion",
                "branches": avail,
                "weights": rec["weights"],
                "threshold": lock_threshold(y_val, rec["val_score"]),
                "val_score": rec["val_score"],
                "ext_score": rec["ext_score"],
                "y_val": y_val,
                "y_ext": y_ext,
                "val_ids": val_scores["ID"].astype(str).tolist(),
                "ext_ids": ext_scores["ID"].astype(str).tolist(),
                "detail": rec["detail"],
                "feature_families": "score_level_fusion",
            }
            val_scores[f"Best_V9_score_fusion::{name}"] = rec["val_score"]
            ext_scores[f"Best_V9_score_fusion::{name}"] = rec["ext_score"]

    artifact = {"artifacts": artifacts, "fit_table": pd.DataFrame(fit_rows), "val_scores": val_scores, "ext_scores": ext_scores, "blocks": blocks}
    with artifact_path.open("wb") as f:
        pickle.dump(artifact, f)
    pd.DataFrame(fit_rows).to_csv(model_dir / "v9_locked_candidate_fit_table.csv", index=False, encoding="utf-8-sig")
    val_scores.to_csv(model_dir / "v9_locked_candidate_val_scores.csv", index=False, encoding="utf-8-sig")
    ext_scores.to_csv(model_dir / "v9_locked_candidate_external_original_scores.csv", index=False, encoding="utf-8-sig")
    return artifact


def score_candidates_on_variant(artifacts_obj: dict, ext_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    artifacts = artifacts_obj["artifacts"]
    y = ext_frame["label_4clf"].astype(int).to_numpy()
    ids = ext_frame["ID"].astype(str).tolist()
    score_df = pd.DataFrame({"ID": ids, "y_true": y})
    failure_rows = []
    branch_scores: dict[str, np.ndarray] = {}
    for method, artifact in artifacts.items():
        if str(method).startswith("Best_V9_score_fusion::") or artifact.get("type") == "score_fusion":
            continue
        try:
            score, missing = score_artifact(artifact, ext_frame)
            score_df[method] = score
            branch_scores[method] = score
            for pid, nmiss in zip(ids, missing):
                if nmiss > 0:
                    failure_rows.append({
                        "patient_id": pid,
                        "phase": "multi",
                        "mask_variant": "set_by_caller",
                        "feature_family": artifact.get("feature_families", ""),
                        "candidate": method,
                        "failure_reason": f"missing_feature_values_before_train_imputation:{int(nmiss)}",
                    })
        except Exception as exc:
            for pid in ids:
                failure_rows.append({
                    "patient_id": pid,
                    "phase": "multi",
                    "mask_variant": "set_by_caller",
                    "feature_family": artifact.get("feature_families", ""),
                    "candidate": method,
                    "failure_reason": f"candidate_score_failed:{exc}",
                })

    for method, artifact in artifacts.items():
        if artifact.get("type") != "score_fusion":
            continue
        branches = artifact["branches"]
        if not all(b in score_df.columns for b in branches):
            failure_rows.append({
                "patient_id": "ALL",
                "phase": "multi",
                "mask_variant": "set_by_caller",
                "feature_family": artifact.get("feature_families", ""),
                "candidate": method,
                "failure_reason": "missing_branch_scores:" + "|".join([b for b in branches if b not in score_df.columns]),
            })
            continue
        x = score_df[branches].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if method.startswith("Best_V9_score_fusion::F1_rank"):
            val = artifacts_obj["val_scores"][branches].to_numpy(dtype=float)
            score = np.zeros(len(score_df), dtype=float)
            for j in range(x.shape[1]):
                sorted_val = np.sort(val[:, j])
                score += np.searchsorted(sorted_val, x[:, j], side="right") / (len(sorted_val) + 1)
            score = score / x.shape[1]
        elif method.startswith("Best_V9_score_fusion::F2_zscore"):
            val = artifacts_obj["val_scores"][branches].to_numpy(dtype=float)
            mu = np.nanmean(val, axis=0)
            sd = np.nanstd(val, axis=0)
            sd = np.where(sd > 1e-8, sd, 1.0)
            score = 1.0 / (1.0 + np.exp(-np.nanmean((x - mu) / sd, axis=1)))
        else:
            weights = artifact.get("weights") or {b: 1.0 / len(branches) for b in branches}
            w = np.asarray([weights.get(b, 0.0) for b in branches], dtype=float)
            if not np.isfinite(w).all() or np.abs(w).sum() <= 0:
                w = np.full(len(branches), 1.0 / len(branches))
            w = w / np.sum(w)
            score = np.nan_to_num(x, nan=np.nanmedian(x, axis=0)) @ w
        score_df[method] = score
    return score_df, pd.DataFrame(failure_rows)


def score_icc_three_reps(mat: np.ndarray) -> float:
    x = np.asarray(mat, dtype=float)
    x = x[np.isfinite(x).all(axis=1)]
    if x.shape[0] < 3 or x.shape[1] < 2:
        return np.nan
    n, k = x.shape
    mean_subject = x.mean(axis=1, keepdims=True)
    mean_rater = x.mean(axis=0, keepdims=True)
    grand = x.mean()
    ss_subject = k * ((mean_subject - grand) ** 2).sum()
    ss_rater = n * ((mean_rater - grand) ** 2).sum()
    ss_error = ((x - mean_subject - mean_rater + grand) ** 2).sum()
    ms_subject = ss_subject / max(1, n - 1)
    ms_error = ss_error / max(1, (n - 1) * (k - 1))
    return float((ms_subject - ms_error) / (ms_subject + (k - 1) * ms_error)) if (ms_subject + (k - 1) * ms_error) != 0 else np.nan


def threshold_metrics_for_scores(score_df: pd.DataFrame, artifacts: dict, n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    auc_rows = []
    flip_rows = []
    y = score_df["y_true"].astype(int).to_numpy()
    for method in [c for c in score_df.columns if c not in {"ID", "y_true", "variant"}]:
        artifact = artifacts.get(method)
        if artifact is None:
            artifact = artifacts.get(method.replace("Best_V9_score_fusion::", "Best_V9_score_fusion::"))
        threshold = artifact.get("threshold", 0.5) if isinstance(artifact, dict) else 0.5
        for variant, dfv in score_df.groupby("variant"):
            s = pd.to_numeric(dfv[method], errors="coerce").to_numpy(dtype=float)
            yy = dfv["y_true"].astype(int).to_numpy()
            mask = np.isfinite(s)
            if mask.sum() < 3 or len(np.unique(yy[mask])) < 2:
                continue
            b = bootstrap_auc(yy[mask], s[mask], n_boot=n_boot, seed=seed)
            ops = clinical_ops(yy, s, threshold)
            auc_rows.append({"mode": "external_final_only", "candidate": method, "variant": variant, **b})
            rows.append({"mode": "external_final_only", "candidate": method, "variant": variant, "threshold_locked": threshold, **ops})
        if "original" in set(score_df["variant"]):
            orig = score_df[score_df["variant"] == "original"][["ID", method]].rename(columns={method: "original_score"})
            orig_pred = (orig["original_score"].to_numpy(dtype=float) >= threshold).astype(int)
            for variant in ["dilate_1px", "erode_1px"]:
                cur = score_df[score_df["variant"] == variant][["ID", method]].merge(orig, on="ID", how="inner")
                if cur.empty:
                    continue
                pred = (cur[method].to_numpy(dtype=float) >= threshold).astype(int)
                base = (cur["original_score"].to_numpy(dtype=float) >= threshold).astype(int)
                flip_rows.append({
                    "mode": "external_final_only",
                    "candidate": method,
                    "variant": variant,
                    "n_common": int(len(cur)),
                    "threshold_locked": threshold,
                    "flip_rate": float(np.mean(pred != base)),
                    "classification_consistency": float(np.mean(pred == base)),
                })
    return pd.DataFrame(rows), pd.DataFrame(auc_rows), pd.DataFrame(flip_rows)


def perturbation_master(score_df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    rows = []
    for method in [c for c in score_df.columns if c not in {"ID", "y_true", "variant"}]:
        byv = {v: d[["ID", "y_true", method]].rename(columns={method: f"score_{v}"}) for v, d in score_df.groupby("variant")}
        if "original" not in byv:
            continue
        original = byv["original"]
        y0 = original["y_true"].astype(int).to_numpy()
        s0 = original["score_original"].to_numpy(dtype=float)
        row = {
            "mode": "external_final_only",
            "candidate": method,
            "AUC_original": auc_hcc_icc(y0, s0),
            "n_original": int(np.isfinite(s0).sum()),
            "threshold_locked": artifacts.get(method, {}).get("threshold", 0.5),
        }
        mats = [original[["ID", "score_original"]]]
        for variant in ["dilate_1px", "erode_1px"]:
            if variant not in byv:
                continue
            cur = original.merge(byv[variant], on=["ID", "y_true"], how="inner")
            y = cur["y_true"].astype(int).to_numpy()
            so = cur["score_original"].to_numpy(dtype=float)
            sp = cur[f"score_{variant}"].to_numpy(dtype=float)
            row[f"AUC_{variant.replace('_1px', '')}"] = auc_hcc_icc(y, sp)
            row[f"delta_AUC_{variant.replace('_1px', '')}"] = row[f"AUC_{variant.replace('_1px', '')}"] - auc_hcc_icc(y, so)
            row[f"Spearman_original_vs_{variant}"] = float(spearmanr(so, sp, nan_policy="omit").correlation)
            row[f"Pearson_original_vs_{variant}"] = float(pearsonr(so[np.isfinite(so) & np.isfinite(sp)], sp[np.isfinite(so) & np.isfinite(sp)])[0]) if (np.isfinite(so) & np.isfinite(sp)).sum() > 2 else np.nan
            row[f"mean_abs_score_change_{variant}"] = float(np.nanmean(np.abs(sp - so)))
            row[f"median_abs_score_change_{variant}"] = float(np.nanmedian(np.abs(sp - so)))
            row[f"n_{variant}"] = int(np.isfinite(sp).sum())
            mats.append(byv[variant][["ID", f"score_{variant}"]])
        mat = mats[0]
        for m in mats[1:]:
            mat = mat.merge(m, on="ID", how="inner")
        row["score_ICC"] = score_icc_three_reps(mat.drop(columns=["ID"]).to_numpy(dtype=float)) if mat.shape[1] >= 3 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def feature_family(feature: str) -> str:
    f = str(feature)
    if f.startswith("DYN_"):
        return "V8_dynamic_radiomics"
    if f.startswith("tumor_liver"):
        return "tumor_liver_contrast"
    if f.startswith("tumor_ratio"):
        return "ratio_style_enhancement"
    if f.startswith(("tumor_mean", "tumor_median", "tumor_p90")):
        return "tumor_enhancement"
    if f.startswith("rim_core_dynamic"):
        return "rim_core_dynamic"
    if f.startswith("rim_core"):
        return "core_rim_contrast"
    if "Imc2" in f:
        return "Imc2_anchor"
    return "radiomics"


def feature_icc_rows(feature_tables: dict[str, pd.DataFrame], features: Sequence[str], used_by: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    base = feature_tables.get("original", pd.DataFrame())
    for feat in features:
        frames = []
        for variant, df in feature_tables.items():
            if feat in df.columns:
                frames.append(df[["ID", feat]].rename(columns={feat: variant}))
        if not frames:
            continue
        mat = frames[0]
        for f in frames[1:]:
            mat = mat.merge(f, on="ID", how="outer")
        vals = mat[[c for c in VARIANTS if c in mat.columns]].apply(pd.to_numeric, errors="coerce")
        original = vals["original"].to_numpy(dtype=float) if "original" in vals.columns else np.full(len(vals), np.nan)
        used = [cand for cand, feats in used_by.items() if feat in feats]
        rows.append({
            "feature": feat,
            "family": feature_family(feat),
            "used_by_candidate": "|".join(used),
            "n_patients": int(vals.dropna().shape[0]),
            "ICC": score_icc_three_reps(vals.to_numpy(dtype=float)),
            "CV": float(np.nanstd(vals.to_numpy(dtype=float)) / max(abs(np.nanmean(vals.to_numpy(dtype=float))), EPS)),
            "mean_abs_change": float(np.nanmean(np.abs(vals.sub(original, axis=0).to_numpy(dtype=float)))) if "original" in vals.columns else np.nan,
            "median_abs_change": float(np.nanmedian(np.abs(vals.sub(original, axis=0).to_numpy(dtype=float)))) if "original" in vals.columns else np.nan,
            "pass_085": False,
            "pass_090": False,
            "original_missing_rate": float(vals.get("original", pd.Series(index=vals.index, dtype=float)).isna().mean()),
            "dilate_missing_rate": float(vals.get("dilate_1px", pd.Series(index=vals.index, dtype=float)).isna().mean()),
            "erode_missing_rate": float(vals.get("erode_1px", pd.Series(index=vals.index, dtype=float)).isna().mean()),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["pass_085"] = out["ICC"] >= 0.85
        out["pass_090"] = out["ICC"] >= 0.90
    return out


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = t
    return out


def delong_roc_test(y_true, score_a, score_b) -> dict:
    y = np.asarray(y_true).astype(int)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b) & np.isin(y, [0, 1])
    y, a, b = y[mask], a[mask], b[mask]
    order = (-y).argsort(kind="mergesort")
    preds = np.vstack([a[order], b[order]])
    m = int(y[order].sum())
    n = len(y) - m
    if m <= 1 or n <= 1:
        return {"delong_delta_auc": np.nan, "delong_z": np.nan, "delong_p": np.nan}
    pos = preds[:, :m]
    neg = preds[:, m:]
    tx = np.vstack([_compute_midrank(pos[i]) for i in range(2)])
    ty = np.vstack([_compute_midrank(neg[i]) for i in range(2)])
    tz = np.vstack([_compute_midrank(preds[i]) for i in range(2)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return {"delong_delta_auc": float(diff), "delong_z": np.nan, "delong_p": np.nan}
    z = diff / np.sqrt(var)
    return {"delong_delta_auc": float(diff), "delong_z": float(z), "delong_p": float(2 * (1 - stats.norm.cdf(abs(z))))}


def collect_existing_scores(output_root: Path, validation_root: Path | None = None) -> pd.DataFrame:
    tables = []
    for path in [
        output_root / V9_ROOT_NAME / "reference_scores" / "reference_external_scores.csv",
        output_root / V9_ROOT_NAME / "enhancement_surrogates" / "surrogate_scores.csv",
        output_root / V9_ROOT_NAME / "phase_token" / "phase_token_external_scores.csv",
        output_root / V9_ROOT_NAME / "score_fusion" / "fusion_scores.csv",
    ]:
        if path.exists():
            tables.append(pd.read_csv(path, encoding="utf-8-sig"))
    if validation_root is not None:
        p = validation_root / "models" / "v9_locked_candidate_external_original_scores.csv"
        if p.exists():
            tables.append(pd.read_csv(p, encoding="utf-8-sig"))
    out = pd.DataFrame()
    for df in tables:
        key = ["ID", "y_true"]
        if out.empty:
            out = df.copy()
        else:
            overlap = sorted((set(out.columns) & set(df.columns)) - set(key))
            out = out.merge(df.drop(columns=overlap), on=key, how="inner")
    return out


def run_split_sensitivity(cfg: dict, method_features: list[str], model_kind: str, n_splits: int, seeds: list[int], seed_base: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    blocks = build_original_blocks(cfg)
    labels = blocks["labels"]
    all_internal = labels[(labels["center"] == "internal") & (labels["label_stage1"] == 1) & (labels["label_4clf"].isin([0, 1]))].copy()
    ids = all_internal["ID"].astype(str).to_numpy()
    y = all_internal["label_4clf"].astype(int).to_numpy()
    int_feat = merge_feature_blocks([blocks["surrogate_internal"], blocks["c2_internal"], blocks["dynamic_internal"]])
    ext_feat = merge_feature_blocks([blocks["surrogate_external"], blocks["c2_external"], blocks["dynamic_external"]])
    rng = np.random.default_rng(seed_base)
    rows = []
    feat_rows = []
    threshold_rows = []
    for seed in seeds:
        for split_idx in range(n_splits):
            rs = int(rng.integers(0, 1_000_000))
            tr, va = train_test_split(ids, test_size=0.25, stratify=y, random_state=rs)
            train, val, ext = stage2_frames(labels, set(tr), set(va), int_feat, ext_feat)
            y_train = train["label_4clf"].astype(int).to_numpy()
            pool = [f for f in method_features if f in train.columns and f in ext.columns]
            selected, _ = select_train_features(train, y_train, pool, k=min(16, max(1, len(pool))))
            if not selected:
                selected = pool[: min(12, len(pool))]
            fit = fit_sklearn_model(train, val, ext, selected, model_kind, seed)
            val_mask = val["label_4clf"].isin([0, 1]).to_numpy()
            y_val = val.loc[val_mask, "label_4clf"].astype(int).to_numpy()
            y_ext = ext["label_4clf"].astype(int).to_numpy()
            thr = lock_threshold(y_val, fit["val_score"][val_mask])
            ops = clinical_ops(y_ext, fit["ext_score"], thr)
            rows.append({
                "mode": "strict_inductive",
                "method": "V9_main_replayed",
                "seed": seed,
                "split": split_idx,
                "random_state": rs,
                "train_n": len(train),
                "val_n_hcc_icc": int(val_mask.sum()),
                "val_auc": auc_hcc_icc(y_val, fit["val_score"][val_mask]),
                "external_auc": auc_hcc_icc(y_ext, fit["ext_score"]),
                "threshold": thr,
                "n_features": len(selected),
                "features": "|".join(selected),
                **ops,
            })
            threshold_rows.append({"seed": seed, "split": split_idx, "threshold": thr})
            for feat in selected:
                feat_rows.append({"seed": seed, "split": split_idx, "feature": feat})
    detail = pd.DataFrame(rows)
    freq = pd.DataFrame(feat_rows)
    if not freq.empty:
        freq = freq.groupby("feature").size().reset_index(name="selected_count")
        freq["selected_frequency"] = freq["selected_count"] / max(1, len(detail))
        freq = freq.sort_values("selected_frequency", ascending=False)
    return detail, freq, pd.DataFrame(threshold_rows)


def summarize_split(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows = []
    rows.append({
        "mode": "strict_inductive",
        "method": "V9_main_replayed",
        "n_runs": int(len(detail)),
        "external_auc_mean": float(detail["external_auc"].mean()),
        "external_auc_std": float(detail["external_auc"].std(ddof=1)),
        "external_auc_min": float(detail["external_auc"].min()),
        "external_auc_max": float(detail["external_auc"].max()),
        "val_auc_mean": float(detail["val_auc"].mean()),
        "val_auc_std": float(detail["val_auc"].std(ddof=1)),
        "threshold_mean": float(detail["threshold"].mean()),
        "threshold_std": float(detail["threshold"].std(ddof=1)),
        "sensitivity_mean": float(detail["sensitivity"].mean()),
        "specificity_mean": float(detail["specificity"].mean()),
        "balanced_accuracy_mean": float(detail["balanced_accuracy"].mean()),
        "V8_reference_external_auc_mean": 0.5894,
        "V8_reference_external_auc_std": 0.090,
    })
    return pd.DataFrame(rows)
