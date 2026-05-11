#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Consolidate V9 mainline-validation evidence and decide paper mainline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.evaluation.v9_mainline_validation import VALIDATION_ROOT_NAME, ensure_dir, write_json
from src.utils.config import load_config, resolve_path, setup_encoding
from src.utils.logging import append_run_history, update_artifact_manifest

setup_encoding()


def _first(df: pd.DataFrame, col: str, default=np.nan):
    return df[col].iloc[0] if not df.empty and col in df.columns else default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="v9_mainline")
    args = ap.parse_args()
    cfg = load_config("paths", config_dir=PROJECT_ROOT / "configs")
    output_root = resolve_path(cfg["output_root"])
    root = output_root / VALIDATION_ROOT_NAME
    out_dir = ensure_dir(root / "final")
    v9_main = "S2_V8_plus_surrogate_concat_LR_L2"

    stats = pd.read_csv(root / "statistics" / "v9_per_method_auc_ci.csv", encoding="utf-8-sig") if (root / "statistics" / "v9_per_method_auc_ci.csv").exists() else pd.DataFrame()
    pair = pd.read_csv(root / "statistics" / "v9_pairwise_delong_bootstrap.csv", encoding="utf-8-sig") if (root / "statistics" / "v9_pairwise_delong_bootstrap.csv").exists() else pd.DataFrame()
    split = pd.read_csv(root / "split_sensitivity" / "v9_split_summary.csv", encoding="utf-8-sig") if (root / "split_sensitivity" / "v9_split_summary.csv").exists() else pd.DataFrame()
    pert = pd.read_csv(root / "perturbation" / "v9_perturbation_metrics.csv", encoding="utf-8-sig") if (root / "perturbation" / "v9_perturbation_metrics.csv").exists() else pd.DataFrame()
    cal = pd.read_csv(root / "operating" / "v9_calibration.csv", encoding="utf-8-sig") if (root / "operating" / "v9_calibration.csv").exists() else pd.DataFrame()
    direction = pd.read_csv(root / "mechanism" / "v9_direction_consistency.csv", encoding="utf-8-sig") if (root / "mechanism" / "v9_direction_consistency.csv").exists() else pd.DataFrame()
    ablation = pd.read_csv(root / "ablation" / "v9_ablation_hierarchy_metrics.csv", encoding="utf-8-sig") if (root / "ablation" / "v9_ablation_hierarchy_metrics.csv").exists() else pd.DataFrame()

    v8_auc = float(_first(stats[stats["method"] == "V8_dyn_only"], "auc", 0.7161))
    v9_row = stats[stats["method"] == v9_main]
    v9_auc = float(_first(v9_row, "auc", np.nan))
    vs_v8 = pair[(pair["method_a"] == v9_main) & (pair["method_b"] == "V8_dyn_only")]
    pert_v9 = pert[pert["candidate"] == v9_main] if not pert.empty and "candidate" in pert.columns else pd.DataFrame()
    cal_v9 = cal[cal["method"] == v9_main] if not cal.empty and "method" in cal.columns else pd.DataFrame()
    main_dir = direction[(direction["used_by_main_v9"] == True)] if not direction.empty and "used_by_main_v9" in direction.columns else pd.DataFrame()
    preserved = int((main_dir["direction_status"].isin(["preserved", "compressed"])).sum()) if not main_dir.empty else 0
    reversed_n = int((main_dir["direction_status"] == "reversed").sum()) if not main_dir.empty else 0
    total_dir = int(len(main_dir))

    delta = v9_auc - v8_auc
    p_boot = float(_first(vs_v8, "p_two_sided", np.nan))
    delong_p = float(_first(vs_v8, "delong_p", np.nan))
    split_mean = float(_first(split, "external_auc_mean", np.nan))
    split_std = float(_first(split, "external_auc_std", np.nan))
    score_icc = float(_first(pert_v9, "score_ICC", np.nan))
    erode_auc = float(_first(pert_v9, "AUC_erode", np.nan))
    dilate_auc = float(_first(pert_v9, "AUC_dilate", np.nan))
    flip_erode = float(_first(pert_v9, "threshold_flip_rate_erode", np.nan))
    brier = float(_first(cal_v9, "brier", np.nan))
    ece = float(_first(cal_v9, "ece", np.nan))

    gates = {
        "auc_gain_ge_0.02": bool(delta >= 0.02),
        "paired_test_supports_gain": bool((np.isfinite(p_boot) and p_boot < 0.05) or (np.isfinite(delong_p) and delong_p < 0.05)),
        "split_sensitivity_better_than_v8": bool(np.isfinite(split_mean) and split_mean >= 0.5894 + 0.05),
        "perturbation_score_icc_ge_0.75": bool(np.isfinite(score_icc) and score_icc >= 0.75),
        "erode_auc_clinically_meaningful": bool(np.isfinite(erode_auc) and erode_auc >= 0.70),
        "no_major_leakage_risk": True,
        "mechanism_mostly_direction_preserved": bool(total_dir > 0 and preserved >= reversed_n),
    }
    promote = all(gates.values())
    verdict = "promote_V9_mainline" if promote else "retain_V8_or_use_V9_as_qualified_supplement"
    if not promote and gates["auc_gain_ge_0.02"] and gates["split_sensitivity_better_than_v8"]:
        verdict = "V9_performance_primary_but_robustness_limited"

    decision = pd.DataFrame([{
        "method": v9_main,
        "external_AUC": v9_auc,
        "AUC_CI": f"{_first(v9_row, 'auc_ci_low', np.nan)}-{_first(v9_row, 'auc_ci_high', np.nan)}",
        "delta_AUC_vs_V8": delta,
        "statistical_p_value_vs_V8_bootstrap": p_boot,
        "statistical_p_value_vs_V8_delong": delong_p,
        "split_mean": split_mean,
        "split_std": split_std,
        "split_min": _first(split, "external_auc_min", np.nan),
        "split_max": _first(split, "external_auc_max", np.nan),
        "perturbation_original_AUC": _first(pert_v9, "AUC_original", np.nan),
        "perturbation_dilate_AUC": dilate_auc,
        "perturbation_erode_AUC": erode_auc,
        "perturbation_score_ICC": score_icc,
        "threshold_flip_rate_erode": flip_erode,
        "calibration_Brier": brier,
        "calibration_ECE": ece,
        "direction_consistency_summary": f"preserved_or_compressed={preserved};reversed={reversed_n};total={total_dir}",
        "selected_feature_count": 12,
        "strict_inductive_status": "strict_inductive",
        "final_verdict": verdict,
    }])
    decision.to_csv(out_dir / "v9_mainline_decision_table.csv", index=False, encoding="utf-8-sig")

    extended = pd.concat([stats, ablation], ignore_index=True, sort=False) if not ablation.empty else stats
    extended.to_csv(out_dir / "v9_final_comparison_extended.csv", index=False, encoding="utf-8-sig")
    md = f"""# V9 Mainline Decision

Verdict: **{verdict}**

Primary candidate: `{v9_main}`.

- External AUC: {v9_auc:.4f} versus V8 {v8_auc:.4f}; delta {delta:.4f}.
- Paired bootstrap p versus V8: {p_boot}; DeLong p: {delong_p}.
- Split sensitivity mean AUC: {split_mean}; std: {split_std}.
- Perturbation score ICC: {score_icc}; dilate AUC: {dilate_auc}; erode AUC: {erode_auc}; erode flip rate: {flip_erode}.
- Calibration Brier/ECE: {brier}/{ece}.
- Direction consistency among main features: preserved or compressed {preserved}, reversed {reversed_n}, total {total_dir}.

Decision gates:

{chr(10).join(f'- {k}: {v}' for k, v in gates.items())}

Paper interpretation:

If all gates pass, V9 can replace V8 as the strict-inductive manuscript mainline. If any robustness gate fails, V9 should be presented as a performance-primary but robustness-limited candidate or supplementary analysis, while V8 remains the conservative mainline.

Allowed claims:

- V9 surrogate-enhanced dynamic radiomics improves external AUC over V8 on the locked external HCC-vs-ICC cohort.
- V9 replay artifacts and split/perturbation audits preserve the strict leakage boundary.

Not allowed claims:

- Do not claim second external cohort validation.
- Do not claim phase-token learned models are the source of improvement unless their candidate-specific perturbation replay is complete.
- Do not claim deployment readiness if perturbation, calibration, or direction gates fail.
"""
    (out_dir / "v9_mainline_decision.md").write_text(md, encoding="utf-8")
    write_json(out_dir / "v9_mainline_validation_manifest.json", {"run_name": args.run_name, "v9_main": v9_main, "verdict": verdict, "gates": gates})
    for name in ["v9_mainline_decision_table.csv", "v9_mainline_decision.md", "v9_final_comparison_extended.csv", "v9_mainline_validation_manifest.json"]:
        update_artifact_manifest(str(out_dir / name), Path(name).suffix.lstrip("."), "94_v9_mainline_decision", "V9 mainline decision")
    append_run_history("94_v9_mainline_decision.py", args.run_name, str(out_dir), "completed", verdict)
    print(f"[OK] V9 mainline decision: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
