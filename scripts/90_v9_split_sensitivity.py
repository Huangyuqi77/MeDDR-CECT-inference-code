#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Full 3 x 3 split sensitivity for replayed V9 main candidate."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.v9_mainline_validation import (
    VALIDATION_ROOT_NAME,
    ensure_dir,
    fit_locked_candidates,
    run_split_sensitivity,
    summarize_split,
    write_json,
)
from src.utils.config import load_config, resolve_path, setup_encoding
from src.utils.logging import append_run_history, update_artifact_manifest

setup_encoding()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="v9_mainline")
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cfg = load_config("paths", config_dir=PROJECT_ROOT / "configs")
    output_root = resolve_path(cfg["output_root"])
    out_dir = ensure_dir(output_root / VALIDATION_ROOT_NAME / "split_sensitivity")
    locked = fit_locked_candidates(cfg, out_dir, seed=args.seed)
    main = locked["artifacts"]["S2_V8_plus_surrogate_concat_LR_L2"]
    detail, freq, thresholds = run_split_sensitivity(cfg, list(main["features"]), "LR_L2", args.n_splits, args.seeds, args.seed)
    summary = summarize_split(detail)
    detail.to_csv(out_dir / "v9_split_sensitivity_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "v9_split_summary.csv", index=False, encoding="utf-8-sig")
    freq.to_csv(out_dir / "v9_split_feature_frequency.csv", index=False, encoding="utf-8-sig")
    thresholds.to_csv(out_dir / "v9_split_thresholds.csv", index=False, encoding="utf-8-sig")
    write_json(out_dir / "v9_split_manifest.json", {
        "run_name": args.run_name,
        "method": "S2_V8_plus_surrogate_concat_LR_L2",
        "n_runs": int(len(detail)),
        "n_splits": args.n_splits,
        "seeds": args.seeds,
        "mode": "strict_inductive",
        "V8_reference_external_auc_mean": 0.5894,
        "V8_reference_external_auc_std": 0.090,
    })
    for name in ["v9_split_sensitivity_metrics.csv", "v9_split_summary.csv", "v9_split_feature_frequency.csv", "v9_split_thresholds.csv", "v9_split_manifest.json"]:
        update_artifact_manifest(str(out_dir / name), Path(name).suffix.lstrip("."), "90_v9_split_sensitivity", "V9 3x3 internal split sensitivity")
    append_run_history("90_v9_split_sensitivity.py", args.run_name, str(out_dir), "completed", f"runs={len(detail)}")
    print(f"[OK] V9 split sensitivity runs={len(detail)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
