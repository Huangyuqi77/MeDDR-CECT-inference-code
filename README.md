# MeDDR CECT Inference Code

This repository contains the curated code release for the MeDDR manuscript:

**Strict-inductive multiphase CT enhancement modelling for HCC-vs-ICC differentiation**

The release is intentionally code-only. It does not include raw CT images, masks,
feature tables, result CSVs, trained model artifacts, manuscript PDFs, or caches.

Public repository:

https://github.com/Huangyuqi77/MeDDR-CECT-inference-code

## Contents

- `src/features/enhancement_surrogates.py`: mechanism-guided multiphase
  enhancement surrogate construction.
- `src/evaluation/v9_eval.py`: train-only preprocessing, feature selection,
  classifier fitting, locked thresholds, metrics, and dynamic-feature helpers.
- `src/evaluation/v9_mainline_validation.py`: locked MeDDR candidate replay,
  split sensitivity, perturbation rescoring, and audit helpers.
- `src/evaluation/clinical.py`: calibration, bootstrap AUC, decision-curve, and
  operating-point utilities.
- `scripts/80_build_enhancement_surrogate_features.py`: build MeDDR surrogate
  feature tables from radiomics caches and masks.
- `scripts/89_v9_pairwise_stats.py` to `scripts/94_v9_mainline_decision.py`:
  manuscript-facing statistical comparison, split replay, mechanism audit,
  ablation, operating analysis, and final claim-boundary scripts.

## Data

The imaging dataset is not redistributed here. The public source dataset is:

Luo, J. et al. Primary Liver Cancer CECT Imaging Dataset. Science Data Bank.
https://doi.org/10.57760/sciencedb.12207

Dataset paper:

Luo, J. et al. Comprehensive multi-phase 3D contrast-enhanced CT imaging for
primary liver cancer. Scientific Data 12, 768 (2025).
https://doi.org/10.1038/s41597-025-05125-2

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
```

On Linux/macOS, use `.venv/bin/pip` instead of `.venv/Scripts/pip`.

Copy `configs/paths.example.yaml` to `configs/paths.yaml` and fill in local
paths to radiomics caches, split labels, masks, and output directories.

## Reproducibility Boundary

The manuscript model is strict-inductive:

- imputation, scaling, feature selection, model fitting, fusion, and threshold
  locking use internal training/validation data only;
- external labels are used only for final metrics and prespecified audits;
- external feature distributions are not used to fit preprocessors or choose
  the strict primary model.

## Important Exclusions

The following are excluded by `.gitignore` and should not be committed:

- raw imaging data and masks;
- feature caches and generated result tables;
- trained model files (`*.pkl`, `*.joblib`, `*.pt`, `*.pth`);
- manuscript PDFs and figures;
- local path configuration files.

## Citation

If this code is used, cite the associated manuscript and the source dataset.
