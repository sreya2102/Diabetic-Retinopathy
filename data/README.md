# RETINASCAN Dataset Policy & Organization

This directory documents the dataset layout for the Diabetic Retinopathy screening system.

> [!IMPORTANT]
> **Git Policy:**
> Do NOT commit dataset files (images, masks, CSV annotations) to git.
> Dataset files are ignored via `.gitignore`.

## Recommended Directory Structure

Local datasets should be placed in subdirectories configured outside of version control:

```
data/
├── raw/
│   ├── idrid/           # Indian Diabetic Retinopathy Image Dataset
│   ├── eyepacs/         # Kaggle EyePACS Dataset
│   ├── aptos/           # APTOS 2019 Blindness Detection
│   ├── drive/           # Digital Retinal Images for Vessel Extraction
│   └── ddr/             # DeepDR Dataset
├── processed/           # Crop / CLAHE preprocessed image caches
├── annotations/         # Ground truth CSV labels & segmentation masks
└── splits/              # Train / Validation / Test split definitions (JSON/CSV)
```

## Dataset Overview

1. **IDRiD (Indian Diabetic Retinopathy Image Dataset)**:
   - Primary target dataset for DR severity grading, lesion segmentation (MA, HE, EX, SE), and optic disc/fovea locations.
2. **EyePACS / Kaggle DR**:
   - Primary dataset for large-scale DR classification training.
3. **APTOS 2019**:
   - Secondary dataset for DR grading validation in resource-constrained settings.
4. **DRIVE**:
   - Benchmark dataset for blood vessel segmentation validation.
5. **DDR**:
   - Dataset for lesion detection and grading benchmark comparison.

## Path Configuration

Dataset root directory path must be configured dynamically in python modules or via environment variables (`DATA_DIR`). Do not hard-code absolute local paths in code files.
