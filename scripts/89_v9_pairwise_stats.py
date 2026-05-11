#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paired DeLong and bootstrap validation for V9 against key baselines."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.evaluation.v9_eval import clinical_ops
from src.evaluation.v9_mainline_validation import (
    VALIDATION_ROOT_NAME,
    collect_existing_scores,
    delong_roc_test,
    ensure_dir,
    fit_locked_candidates,
    paired_bootstrap_delta,
    write_json,
)
from src.evaluation.clinical import bootstrap_auc
from src.utils.config import load_config, resolve_path, setup_encoding
from src.utils.logging import append_run_history, update_artifact_manifest

setup_encoding()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="v9_mainline")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cfg = load_config("paths", config_dir=PROJECT_ROOT / "configs")
    output_root = resolve_path(cfg["output_root"])
    root = output_root / VALIDATION_ROOT_NAME
    out_dir = ensure_dir(root / "statistics")
    locked = fit_locked_candidates(cfg, out_dir, seed=args.seed)
    scores = collect_existing_scores(output_root, root)
    v9_main = "S2_V8_plus_surrogate_concat_LR_L2"
    if v9_main not in scores.columns:
        raise FileNotFoundError(f"V9 main score column missing: {v9_main}")
    y = scores["y_true"].astype(int).to_numpy()

    methods = ["E0", "Imc2_only", "V8_dyn_only", "V8_fusion", "V7_exploratory", "full_strict_best_peri_fusion"]
    best_fusion = [c for c in scores.columns if c.startswith("Best_V9_score_fusion::F2")]
    if best_fusion:
        methods.append(best_fusion[0])
    methods = [m for m in methods if m in scores.columns]

    pair_rows = []
    for m in methods:
        row = {"mode": "external_final_only", "method_a": v9_main, "method_b": m}
        row.update(delong_roc_test(y, scores[v9_main], scores[m]))
        row.update(paired_bootstrap_delta(y, scores[v9_main], scores[m], args.n_boot, args.seed))
        pair_rows.append(row)

    ci_rows = []
    op_rows = []
    for m in [v9_main] + methods:
        s = scores[m].astype(float).to_numpy()
        ci = bootstrap_auc(y, s, n_boot=args.n_boot, seed=args.seed)
        ci_rows.append({"mode": "external_final_only", "method": m, **ci})
        thr = locked["artifacts"].get(m, {}).get("threshold", 0.5)
        op_rows.append({"mode": "external_final_only", "method": m, "threshold_locked": thr, **clinical_ops(y, s, thr)})

    pd.DataFrame(pair_rows).to_csv(out_dir / "v9_pairwise_delong_bootstrap.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ci_rows).to_csv(out_dir / "v9_per_method_auc_ci.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(op_rows).to_csv(out_dir / "v9_per_method_locked_metrics.csv", index=False, encoding="utf-8-sig")
    write_json(out_dir / "v9_statistical_validation_manifest.json", {
        "run_name": args.run_name,
        "v9_main": v9_main,
        "comparators": methods,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "mode": "external_final_only",
    })
    for name in ["v9_pairwise_delong_bootstrap.csv", "v9_per_method_auc_ci.csv", "v9_per_method_locked_metrics.csv", "v9_statistical_validation_manifest.json"]:
        update_artifact_manifest(str(out_dir / name), Path(name).suffix.lstrip("."), "89_v9_pairwise_stats", "V9 paired statistical validation")
    append_run_history("89_v9_pairwise_stats.py", args.run_name, str(out_dir), "completed", f"comparisons={len(pair_rows)}")
    print(f"[OK] V9 pairwise stats comparisons={len(pair_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
