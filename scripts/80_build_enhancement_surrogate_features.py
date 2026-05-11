#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build V9 mechanism-guided enhancement surrogate features."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.features.enhancement_surrogates import build_enhancement_surrogates
from src.utils.config import load_config, resolve_path, setup_encoding
from src.utils.logging import append_run_history, update_artifact_manifest
from src.evaluation.v9_eval import ensure_dir, load_labels, write_json

setup_encoding()


def stage_ids(labels: pd.DataFrame, cohort: str) -> list[str]:
    if cohort == "internal":
        df = labels[(labels["center"] == "internal") & (labels["label_stage1"] == 1)]
    else:
        df = labels[(labels["center"] == "external") & (labels["label_4clf"].isin([0, 1]))]
    return sorted(df["ID"].astype(str).str.strip().tolist())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="v9_main")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--overwrite-core-rim", action="store_true")
    args = ap.parse_args()

    cfg = load_config("paths", config_dir=PROJECT_ROOT / "configs")
    output_root = resolve_path(cfg["output_root"])
    cache_root = resolve_path(cfg["feature_cache_root"])
    split_dir = resolve_path(cfg["split_root"])
    registered_root = resolve_path(cfg["registered_mask_cache_root"])
    out_dir = ensure_dir(output_root / "v9_dynamic_upgrade" / "enhancement_surrogates")
    labels, _, _ = load_labels(split_dir)

    out_paths = {
        "internal": out_dir / "surrogate_features_internal.csv",
        "external": out_dir / "surrogate_features_external.csv",
    }
    if all(p.exists() for p in out_paths.values()) and not args.overwrite:
        print(f"[CACHE] Surrogate feature tables already exist under {out_dir}")
    else:
        for cohort in ("internal", "external"):
            root = resolve_path(cfg[f"{cohort}_root"])
            ids = stage_ids(labels, cohort)
            features, report = build_enhancement_surrogates(
                cache_root=cache_root,
                cohort=cohort,
                data_root=root,
                registered_root=registered_root,
                out_dir=out_dir,
                ids=ids,
                overwrite_core_rim=args.overwrite_core_rim,
            )
            features = features[features["ID"].astype(str).str.strip().isin(ids)].copy()
            features.to_csv(out_paths[cohort], index=False, encoding="utf-8-sig")
            print(f"[OK] {cohort}: features={features.shape}; core/rim report={report.shape}")

    manifest = {
        "run_name": args.run_name,
        "mode": "strict_inductive_feature_extraction_label_free",
        "inputs": {
            "radiomics_cache": str(cache_root),
            "registered_mask_cache": str(registered_root),
            "internal_root": cfg["internal_root"],
            "external_root": cfg["external_root"],
        },
        "outputs": {k: str(v) for k, v in out_paths.items()},
        "feature_rule": "tumor/liver multiphase first-order enhancement surrogates plus tumor-internal core-rim contrast",
    }
    for cohort, path in out_paths.items():
        df = pd.read_csv(path, encoding="utf-8-sig")
        manifest[f"{cohort}_shape"] = list(df.shape)
        report_p = out_dir / f"core_rim_case_report_{cohort}.csv"
        if report_p.exists():
            rep = pd.read_csv(report_p, encoding="utf-8-sig")
            manifest[f"{cohort}_core_rim_failures"] = int((rep["status"] != "ok").sum()) if "status" in rep.columns else None
    write_json(out_dir / "surrogate_manifest.json", manifest)
    update_artifact_manifest(str(out_dir / "surrogate_features_internal.csv"), "csv", "80_build_enhancement_surrogate_features", "V9 internal enhancement surrogate features", True)
    update_artifact_manifest(str(out_dir / "surrogate_features_external.csv"), "csv", "80_build_enhancement_surrogate_features", "V9 external enhancement surrogate features", True)
    append_run_history("80_build_enhancement_surrogate_features.py", "P/C1/C2/C3 tumor+liver radiomics caches and tumor masks", str(out_dir), "completed", str(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
