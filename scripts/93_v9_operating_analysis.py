#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paper-facing V9 operating metrics, calibration, and DCA."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.evaluation.clinical import calibration_metrics, decision_curve
from src.evaluation.v9_eval import clinical_ops
from src.evaluation.v9_mainline_validation import VALIDATION_ROOT_NAME, collect_existing_scores, ensure_dir, fit_locked_candidates, write_json
from src.utils.config import load_config, resolve_path, setup_encoding
from src.utils.logging import append_run_history, update_artifact_manifest

setup_encoding()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="v9_mainline")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cfg = load_config("paths", config_dir=PROJECT_ROOT / "configs")
    output_root = resolve_path(cfg["output_root"])
    root = output_root / VALIDATION_ROOT_NAME
    out_dir = ensure_dir(root / "operating")
    locked = fit_locked_candidates(cfg, out_dir, seed=args.seed)
    scores = collect_existing_scores(output_root, root)
    final_cmp = pd.read_csv(output_root / "v9_dynamic_upgrade" / "final" / "v9_final_comparison.csv", encoding="utf-8-sig")
    threshold_map = {str(r["method"]): float(r["threshold_locked"]) for _, r in final_cmp.dropna(subset=["threshold_locked"]).iterrows() if "method" in r}
    for m, a in locked["artifacts"].items():
        if "threshold" in a:
            threshold_map[m] = float(a["threshold"])
    methods = ["E0", "Imc2_only", "V8_dyn_only", "S2_V8_plus_surrogate_concat_LR_L2", "V7_exploratory", "full_strict_best_peri_fusion"]
    methods = [m for m in methods if m in scores.columns]
    y = scores["y_true"].astype(int).to_numpy()
    rows, cal_rows, cal_curve_rows, dca_rows, ppv_rows, trade_rows = [], [], [], [], [], []
    for m in methods:
        s = scores[m].astype(float).to_numpy()
        thr = threshold_map.get(m, 0.5)
        rows.append({"mode": "external_final_only", "method": m, "threshold_locked": thr, **clinical_ops(y, s, thr)})
        cal, curve = calibration_metrics(y, s)
        cal.update({"mode": "external_final_only", "method": m})
        cal_rows.append(cal)
        curve.insert(0, "method", m)
        cal_curve_rows.append(curve)
        d = decision_curve(y, s)
        d.insert(0, "method", m)
        dca_rows.append(d)
        for t in np.linspace(0.05, 0.95, 19):
            ops = clinical_ops(y, s, float(t))
            ppv_rows.append({"method": m, "threshold": float(t), "ppv": ops["ppv"], "npv": ops["npv"], "sensitivity": ops["sensitivity"], "specificity": ops["specificity"]})
        for t in sorted(set([thr, 0.3, 0.5, 0.7])):
            ops = clinical_ops(y, s, float(t))
            trade_rows.append({"method": m, "threshold": float(t), **ops})

    pd.DataFrame(rows).to_csv(out_dir / "v9_operating_metrics_table.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(trade_rows).to_csv(out_dir / "v9_threshold_tradeoff.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cal_rows).to_csv(out_dir / "v9_calibration.csv", index=False, encoding="utf-8-sig")
    pd.concat(cal_curve_rows, ignore_index=True).to_csv(out_dir / "v9_calibration_curve.csv", index=False, encoding="utf-8-sig")
    pd.concat(dca_rows, ignore_index=True).to_csv(out_dir / "v9_decision_curve.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ppv_rows).to_csv(out_dir / "v9_ppv_npv_threshold_curve.csv", index=False, encoding="utf-8-sig")
    write_json(out_dir / "v9_operating_manifest.json", {"run_name": args.run_name, "methods": methods, "mode": "external_final_only"})
    for name in ["v9_operating_metrics_table.csv", "v9_threshold_tradeoff.csv", "v9_calibration.csv", "v9_calibration_curve.csv", "v9_decision_curve.csv", "v9_ppv_npv_threshold_curve.csv", "v9_operating_manifest.json"]:
        update_artifact_manifest(str(out_dir / name), Path(name).suffix.lstrip("."), "93_v9_operating_analysis", "V9 operating calibration DCA")
    append_run_history("93_v9_operating_analysis.py", args.run_name, str(out_dir), "completed", f"methods={len(methods)}")
    print(f"[OK] V9 operating methods={len(methods)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
