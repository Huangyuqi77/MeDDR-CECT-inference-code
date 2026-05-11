"""Mechanism-guided multiphase enhancement surrogate features.

The features in this module are intentionally low-dimensional and
strictly label-free. They are built from existing phase tumor/liver
radiomics caches plus optional tumor-internal core/rim intensity summaries.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.evaluation.v9_eval import EPS, PHASES, load_raw_features

STAT_STEMS = {
    "mean": "original_firstorder_Mean",
    "median": "original_firstorder_Median",
    "p90": "original_firstorder_90Percentile",
}


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


def build_surrogate_from_radiomics(cache_root: Path, cohort: str) -> pd.DataFrame:
    """Build tumor enhancement and tumor-liver contrast features.

    No labels are read here. Missing phases/features remain NaN and are handled
    later by train-only imputers.
    """
    tumor = {ph: load_raw_features(cache_root, ph, "tumor", cohort) for ph in PHASES}
    liver = {ph: load_raw_features(cache_root, ph, "liver", cohort) for ph in PHASES}
    ids = _ids_from_tables({**{f"tumor_{k}": v for k, v in tumor.items()}, **{f"liver_{k}": v for k, v in liver.items()}})
    out = pd.DataFrame({"ID": ids})

    tvals: dict[tuple[str, str], np.ndarray] = {}
    lmean: dict[str, np.ndarray] = {}
    for ph in PHASES:
        for stat, stem in STAT_STEMS.items():
            tvals[(ph, stat)] = _phase_value(tumor, ph, stem, ids)
        lmean[ph] = _phase_value(liver, ph, STAT_STEMS["mean"], ids)

    def add_delta(prefix: str, stat: str, pairs: Sequence[tuple[str, str]]) -> None:
        for a, b in pairs:
            out[f"{prefix}_{a}_minus_{b}"] = tvals[(a, stat)] - tvals[(b, stat)]

    add_delta("tumor_mean", "mean", [("C1", "P"), ("C2", "C1"), ("C3", "C2"), ("C2", "P"), ("C3", "P")])
    add_delta("tumor_median", "median", [("C1", "P"), ("C2", "C1"), ("C3", "C2")])
    add_delta("tumor_p90", "p90", [("C1", "P"), ("C2", "C1"), ("C3", "C2")])

    for a, b in [("C1", "P"), ("C2", "C1"), ("C3", "C2"), ("C2", "P"), ("C3", "P")]:
        out[f"tumor_ratio_{a}_{b}"] = _safe_ratio(tvals[(a, "mean")] - tvals[(b, "mean")], tvals[(b, "mean")])

    contrast = {}
    for ph in PHASES:
        out[f"tumor_liver_mean_ratio_{ph}"] = _safe_ratio(tvals[(ph, "mean")], lmean[ph])
        contrast[ph] = tvals[(ph, "mean")] - lmean[ph]
    for a, b in [("C1", "P"), ("C2", "C1"), ("C3", "C2"), ("C2", "P"), ("C3", "P")]:
        out[f"tumor_liver_delta_{a}_{b}"] = contrast[a] - contrast[b]
    return out


def image_path(root: Path, phase: str, patient_id: str) -> Path:
    return root / phase / "images" / f"{patient_id}.nii.gz"


def mask_path(root: Path, registered_root: Path, cohort: str, phase: str, patient_id: str) -> Path:
    raw = root / phase / "masks" / f"{patient_id}.nii.gz"
    if raw.exists():
        return raw
    reg = registered_root / phase / "masks" / f"{patient_id}.nii.gz"
    return reg


def compute_core_rim_stats_for_ids(
    ids: Sequence[str],
    cohort: str,
    data_root: Path,
    registered_root: Path,
    cache_path: Path,
    report_path: Path,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute tumor-internal core/rim means with resume-friendly CSV cache."""
    import SimpleITK as sitk

    completed: set[tuple[str, str]] = set()
    rows: dict[str, dict] = {}
    reports = []
    if cache_path.exists() and report_path.exists() and not overwrite:
        cached = pd.read_csv(cache_path, encoding="utf-8-sig")
        rep = pd.read_csv(report_path, encoding="utf-8-sig")
        for _, r in rep.iterrows():
            if str(r.get("status")) == "ok":
                completed.add((str(r["ID"]), str(r["phase"])))
        rows = {str(r["ID"]): r.to_dict() for _, r in cached.iterrows()}
        reports = rep.to_dict("records")

    for patient_id in [str(x).strip() for x in ids]:
        rec = rows.get(patient_id, {"ID": patient_id})
        rim_core_by_phase = {}
        for ph in PHASES:
            if (patient_id, ph) in completed:
                continue
            img_p = image_path(data_root, ph, patient_id)
            msk_p = mask_path(data_root, registered_root, cohort, ph, patient_id)
            try:
                if not img_p.exists():
                    raise FileNotFoundError(f"image_missing:{img_p}")
                if not msk_p.exists():
                    raise FileNotFoundError(f"mask_missing:{msk_p}")
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
                core_mean = float(np.nanmean(arr[core]))
                rim_mean = float(np.nanmean(arr[rim]))
                delta = rim_mean - core_mean
                rec[f"rim_core_mean_delta_{ph}"] = delta
                rim_core_by_phase[ph] = delta
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

        # Periodic flush avoids losing long-run progress.
        if len(reports) % 50 == 0:
            pd.DataFrame(rows.values()).to_csv(cache_path, index=False, encoding="utf-8-sig")
            pd.DataFrame(reports).to_csv(report_path, index=False, encoding="utf-8-sig")

    out = pd.DataFrame(rows.values())
    for a, b in [("C1", "P"), ("C2", "C1"), ("C3", "C2"), ("C3", "C1")]:
        out[f"rim_core_dynamic_{a}_minus_{b}"] = pd.to_numeric(out.get(f"rim_core_mean_delta_{a}"), errors="coerce") - pd.to_numeric(out.get(f"rim_core_mean_delta_{b}"), errors="coerce")
    rep_df = pd.DataFrame(reports)
    if not rep_df.empty and {"ID", "phase"}.issubset(rep_df.columns):
        rep_df = rep_df.drop_duplicates(["ID", "phase"], keep="last").sort_values(["ID", "phase"]).reset_index(drop=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False, encoding="utf-8-sig")
    rep_df.to_csv(report_path, index=False, encoding="utf-8-sig")
    return out, rep_df


def build_enhancement_surrogates(
    cache_root: Path,
    cohort: str,
    data_root: Path,
    registered_root: Path,
    out_dir: Path,
    ids: Iterable[str] | None = None,
    overwrite_core_rim: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = build_surrogate_from_radiomics(cache_root, cohort)
    target_ids = sorted(set(base["ID"].astype(str).str.strip()) if ids is None else {str(x).strip() for x in ids})
    core, report = compute_core_rim_stats_for_ids(
        target_ids,
        cohort=cohort,
        data_root=data_root,
        registered_root=registered_root,
        cache_path=out_dir / f"core_rim_stats_{cohort}.csv",
        report_path=out_dir / f"core_rim_case_report_{cohort}.csv",
        overwrite=overwrite_core_rim,
    )
    merged = base.merge(core, on="ID", how="left")
    return merged, report
