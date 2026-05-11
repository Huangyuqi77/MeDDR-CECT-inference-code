#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V9 ablation hierarchy and stable-subset candidate selection."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.evaluation.score_fusion import fit_score_fusions
from src.evaluation.v9_eval import calibration_row, fit_sklearn_model, metric_row, select_train_features, summarize_feature_families, y_arrays
from src.evaluation.v9_mainline_validation import VALIDATION_ROOT_NAME, ensure_dir, feature_family, fit_locked_candidates, write_json
from src.utils.config import load_config, resolve_path, setup_encoding
from src.utils.logging import append_run_history, update_artifact_manifest

setup_encoding()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="v9_mainline")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    cfg = load_config("paths", config_dir=PROJECT_ROOT / "configs")
    output_root = resolve_path(cfg["output_root"])
    root = output_root / VALIDATION_ROOT_NAME
    out_dir = ensure_dir(root / "ablation")
    locked = fit_locked_candidates(cfg, out_dir, seed=args.seed)
    train, val, ext = locked["blocks"]["train"], locked["blocks"]["val"], locked["blocks"]["ext"]
    y_train, _, y_ext = y_arrays(train, val, ext)
    val_mask = val["label_4clf"].isin([0, 1]).to_numpy()
    y_val = val.loc[val_mask, "label_4clf"].astype(int).to_numpy()

    all_features = sorted({f for a in locked["artifacts"].values() for f in a.get("features", []) if isinstance(f, str)})
    v8 = [f for f in locked["artifacts"]["V8_dyn_only"]["features"] if f in train.columns]
    imc2 = [f for f in locked["artifacts"]["Imc2_only"]["features"] if f in train.columns]
    surrogate = [f for f in all_features if feature_family(f) not in {"V8_dynamic_radiomics", "Imc2_anchor"}]
    enhancement = [f for f in surrogate if feature_family(f) in {"tumor_enhancement", "ratio_style_enhancement"}]
    ratios = [f for f in surrogate if feature_family(f) == "tumor_liver_contrast"]
    core = [f for f in surrogate if feature_family(f) in {"core_rim_contrast", "rim_core_dynamic"}]

    icc = pd.read_csv(root / "feature_icc" / "v9_feature_perturbation_icc.csv", encoding="utf-8-sig") if (root / "feature_icc" / "v9_feature_perturbation_icc.csv").exists() else pd.DataFrame()
    direction = pd.read_csv(root / "mechanism" / "v9_direction_consistency.csv", encoding="utf-8-sig") if (root / "mechanism" / "v9_direction_consistency.csv").exists() else pd.DataFrame()
    surrogate_set = set(surrogate)
    stable = [f for f in (icc.loc[icc["ICC"] >= 0.85, "feature"].astype(str).tolist() if not icc.empty else surrogate) if f in surrogate_set]
    dir_consistent = [f for f in (direction.loc[direction["direction_status"].isin(["preserved", "compressed"]), "feature"].astype(str).tolist() if not direction.empty else surrogate) if f in surrogate_set]
    stable_dir = sorted(set(stable) & set(dir_consistent))

    candidate_pools = {
        "V8_dyn_only": v8,
        "surrogate_only": surrogate,
        "enhancement_only": enhancement,
        "tumor_liver_ratio_only": ratios,
        "core_rim_only": core,
        "V8_plus_enhancement": sorted(set(v8 + enhancement)),
        "V8_plus_tumor_liver_ratio": sorted(set(v8 + ratios)),
        "V8_plus_core_rim": sorted(set(v8 + core)),
        "V8_plus_all_surrogate": sorted(set(v8 + surrogate)),
        "V8_plus_stable_surrogate_only": sorted(set(v8 + stable)),
        "V8_plus_direction_consistent_surrogate_only": sorted(set(v8 + dir_consistent)),
        "V8_plus_stable_and_direction_consistent_surrogate": sorted(set(v8 + stable_dir)),
        "Imc2_plus_V8_plus_stable_surrogate": sorted(set(imc2 + v8 + stable)),
    }

    metric_rows = []
    auc_rows = []
    cal_rows = []
    val_scores = pd.DataFrame({"ID": val.loc[val_mask, "ID"].astype(str).tolist(), "y_true": y_val})
    ext_scores = pd.DataFrame({"ID": ext["ID"].astype(str).tolist(), "y_true": y_ext})
    for name, pool in candidate_pools.items():
        pool = [f for f in pool if f in train.columns and f in ext.columns]
        if not pool:
            continue
        selected, _ = select_train_features(train, y_train, pool, k=min(16, len(pool)))
        if not selected:
            selected = pool[: min(12, len(pool))]
        fit = fit_sklearn_model(train, val, ext, selected, "LR_L2", args.seed)
        sv, se = fit["val_score"][val_mask], fit["ext_score"]
        extra = {"candidate_group": name, "model_kind": "LR_L2", "model_tag": fit["model_tag"], "n_features": len(selected), "features": "|".join(selected), "feature_families": summarize_feature_families(selected)}
        row = metric_row(name, "strict_inductive", y_val, sv, y_ext, se, args.n_boot, args.seed, extra)
        metric_rows.append(row)
        auc_rows.append({k: row.get(k) for k in ["mode", "method", "auc", "auc_ci_low", "auc_ci_high", "candidate_group"]})
        cal, _ = calibration_row(name, "strict_inductive", y_ext, se)
        cal_rows.append(cal)
        val_scores[name] = sv
        ext_scores[name] = se

    # Stable score fusions.
    fusion_methods = [m for m in ["V8_dyn_only", "Imc2_plus_V8_plus_stable_surrogate", "V8_plus_stable_surrogate_only"] if m in val_scores.columns]
    fusions = fit_score_fusions(val_scores[["ID", "y_true"] + fusion_methods], ext_scores[["ID", "y_true"] + fusion_methods], fusion_methods, seed=args.seed) if len(fusion_methods) >= 2 else {}
    for key in ["F1_rank_normalized_average", "F4_shrinkage_stacking"]:
        if key in fusions:
            rec = fusions[key]
            name = f"{key}(V8,Imc2,stable_surrogate)"
            row = metric_row(name, "strict_inductive", y_val, rec["val_score"], y_ext, rec["ext_score"], args.n_boot, args.seed, {"candidate_group": "score_fusion", "model_kind": "score_fusion", "n_features": len(fusion_methods), "features": "|".join(fusion_methods), "feature_families": "score_level_fusion"})
            metric_rows.append(row)
            auc_rows.append({k: row.get(k) for k in ["mode", "method", "auc", "auc_ci_low", "auc_ci_high", "candidate_group"]})
            cal, _ = calibration_row(name, "strict_inductive", y_ext, rec["ext_score"])
            cal_rows.append(cal)

    metrics = pd.DataFrame(metric_rows)
    pert = pd.read_csv(root / "perturbation" / "v9_perturbation_metrics.csv", encoding="utf-8-sig") if (root / "perturbation" / "v9_perturbation_metrics.csv").exists() else pd.DataFrame()
    if not pert.empty:
        metrics = metrics.merge(pert[["candidate", "score_ICC"]].rename(columns={"candidate": "method", "score_ICC": "perturbation_score_ICC"}), on="method", how="left")
    split = pd.read_csv(root / "split_sensitivity" / "v9_split_summary.csv", encoding="utf-8-sig") if (root / "split_sensitivity" / "v9_split_summary.csv").exists() else pd.DataFrame()
    if not split.empty:
        metrics["split_mean_external_auc"] = float(split["external_auc_mean"].iloc[0])
    metrics.to_csv(out_dir / "v9_ablation_hierarchy_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(auc_rows).to_csv(out_dir / "v9_ablation_auc_ci.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(out_dir / "v9_ablation_locked_metrics.csv", index=False, encoding="utf-8-sig")
    write_json(out_dir / "v9_ablation_manifest.json", {"run_name": args.run_name, "n_candidates": int(len(metrics)), "stable_feature_count": len(stable), "direction_consistent_feature_count": len(dir_consistent), "mode": "strict_inductive"})
    for name in ["v9_ablation_hierarchy_metrics.csv", "v9_ablation_auc_ci.csv", "v9_ablation_locked_metrics.csv", "v9_ablation_manifest.json"]:
        update_artifact_manifest(str(out_dir / name), Path(name).suffix.lstrip("."), "92_v9_ablation_hierarchy", "V9 stable-subset ablation hierarchy")
    append_run_history("92_v9_ablation_hierarchy.py", args.run_name, str(out_dir), "completed", f"candidates={len(metrics)}")
    print(f"[OK] V9 ablation candidates={len(metrics)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
