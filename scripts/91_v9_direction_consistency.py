#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V9 feature direction-consistency and mechanism audit."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from src.evaluation.v9_eval import cohen_d
from src.evaluation.v9_mainline_validation import VALIDATION_ROOT_NAME, ensure_dir, feature_family, fit_locked_candidates, write_json
from src.utils.config import load_config, resolve_path, setup_encoding
from src.utils.logging import append_run_history, update_artifact_manifest

setup_encoding()


def _vals(df, feat, label):
    return pd.to_numeric(df.loc[df["label_4clf"].astype(int) == label, feat], errors="coerce").dropna().to_numpy(dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="v9_mainline")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cfg = load_config("paths", config_dir=PROJECT_ROOT / "configs")
    output_root = resolve_path(cfg["output_root"])
    out_dir = ensure_dir(output_root / VALIDATION_ROOT_NAME / "mechanism")
    locked = fit_locked_candidates(cfg, out_dir, seed=args.seed)
    blocks = locked["blocks"]
    train = blocks["train"]
    ext = blocks["ext"]
    main_feats = list(locked["artifacts"]["S2_V8_plus_surrogate_concat_LR_L2"]["features"])
    all_feats = sorted({f for a in locked["artifacts"].values() for f in a.get("features", []) if isinstance(f, str)})

    rows = []
    dist_rows = []
    for feat in all_feats:
        if feat not in train.columns or feat not in ext.columns:
            rows.append({"feature": feat, "family": feature_family(feat), "direction_status": "non-computable", "used_by_main_v9": feat in main_feats})
            continue
        ih, ii = _vals(train, feat, 0), _vals(train, feat, 1)
        eh, ei = _vals(ext, feat, 0), _vals(ext, feat, 1)
        d_int = cohen_d(ih, ii)
        d_ext = cohen_d(eh, ei)
        sign_match = bool(np.sign(d_int) == np.sign(d_ext)) if np.isfinite(d_int) and np.isfinite(d_ext) else False
        if not np.isfinite(d_int) or not np.isfinite(d_ext):
            status = "non-computable"
        elif not sign_match:
            status = "reversed"
        elif abs(d_ext) < 0.25 * max(abs(d_int), 1e-9):
            status = "compressed"
        elif abs(d_ext) < 0.1:
            status = "unstable"
        else:
            status = "preserved"
        try:
            p_int = mannwhitneyu(ih, ii, alternative="two-sided").pvalue
        except Exception:
            p_int = np.nan
        try:
            p_ext = mannwhitneyu(eh, ei, alternative="two-sided").pvalue
        except Exception:
            p_ext = np.nan
        rows.append({
            "feature": feat,
            "family": feature_family(feat),
            "internal_HCC_mean": np.nanmean(ih) if len(ih) else np.nan,
            "internal_ICC_mean": np.nanmean(ii) if len(ii) else np.nan,
            "external_HCC_mean": np.nanmean(eh) if len(eh) else np.nan,
            "external_ICC_mean": np.nanmean(ei) if len(ei) else np.nan,
            "internal_cohen_d": d_int,
            "external_cohen_d": d_ext,
            "sign_match": sign_match,
            "direction_status": status,
            "MWU_p_internal": p_int,
            "MWU_p_external": p_ext,
            "used_by_main_v9": feat in main_feats,
        })
        for cohort, a, b in [("internal", ih, ii), ("external", eh, ei)]:
            dist_rows.append({"feature": feat, "family": feature_family(feat), "cohort": cohort, "HCC_median": np.nanmedian(a) if len(a) else np.nan, "ICC_median": np.nanmedian(b) if len(b) else np.nan, "HCC_n": len(a), "ICC_n": len(b)})

    df = pd.DataFrame(rows)
    dist = pd.DataFrame(dist_rows)
    main = df[df["used_by_main_v9"] == True]
    preserved = int((main["direction_status"] == "preserved").sum()) if not main.empty else 0
    reversed_n = int((main["direction_status"] == "reversed").sum()) if not main.empty else 0
    summary = f"""# V9 Mechanism Summary

Main replayed V9 candidate: `S2_V8_plus_surrogate_concat_LR_L2`.

Selected main-feature direction status: preserved={preserved}, reversed={reversed_n}, total={len(main)}.

Interpretation: V9 gain is considered mechanism-supported when most high-impact enhancement and tumor-liver features are preserved or compressed rather than reversed. Core-rim features with reversal are treated as robustness concerns and are considered by the stable-subset ablation.
"""
    df.to_csv(out_dir / "v9_direction_consistency.csv", index=False, encoding="utf-8-sig")
    dist.to_csv(out_dir / "v9_feature_distribution_summary.csv", index=False, encoding="utf-8-sig")
    (out_dir / "v9_mechanism_summary.md").write_text(summary, encoding="utf-8")
    write_json(out_dir / "v9_mechanism_manifest.json", {"run_name": args.run_name, "features": int(len(df)), "main_features": int(len(main)), "mode": "external_final_only"})
    for name in ["v9_direction_consistency.csv", "v9_feature_distribution_summary.csv", "v9_mechanism_summary.md", "v9_mechanism_manifest.json"]:
        update_artifact_manifest(str(out_dir / name), Path(name).suffix.lstrip("."), "91_v9_direction_consistency", "V9 direction-consistency mechanism audit")
    append_run_history("91_v9_direction_consistency.py", args.run_name, str(out_dir), "completed", f"features={len(df)}")
    print(f"[OK] V9 direction consistency features={len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
