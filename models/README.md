# RETINASCAN Model Artifacts & Management

This directory stores trained model weights, checkpoints, and model metadata.

> [!IMPORTANT]
> **Git Policy:**
> Model weights (`*.pt`, `*.pth`, `*.onnx`, `*.h5`) must NOT be committed to git.
> Large binary files are excluded in `.gitignore`.

## Model Metadata Schema

For each trained model, a corresponding JSON metadata file should be saved alongside local weight files:

```json
{
  "model_name": "dr_classifier_efficientnet_b0",
  "version": "1.0.0",
  "architecture": "EfficientNet-B0",
  "input_size": [512, 512, 3],
  "num_classes": 5,
  "training_dataset": "IDRiD + EyePACS sample",
  "preprocessing_version": "v1.0-clahe-crop",
  "metrics": {
    "accuracy": 0.0,
    "quadratic_weighted_kappa": 0.0,
    "referable_sensitivity": 0.0,
    "referable_specificity": 0.0
  },
  "created_at": "2026-08-24"
}
```

## Phase 1 Status
No model weights are included in Phase 1. Neural network components currently define architectural interfaces and inference contracts only.
