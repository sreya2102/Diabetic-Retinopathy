# RETINASCAN: UI Architecture & Integration Contract

**Project Title:** Explainable AI for Diabetic Retinopathy Screening in Rural India  
**Branch:** `feature/ui-dashboard`  
**Application Name:** `RETINASCAN`  

---

## 1. Architectural Overview

RETINASCAN is a clinical-grade web application built in Python using Streamlit, Plotly, ReportLab, and OpenCV. It is engineered to support community-scale diabetic retinopathy screening across rural Primary Health Centres (PHCs) with low-bandwidth tele-connectivity.

```
┌─────────────────────────────────────────────────────────────┐
│                    RETINASCAN Streamlit UI                  │
│  app/main.py (Multi-Page Controller, Theme & Navigation)    │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
       ┌───────▼────────┐             ┌───────▼────────┐
       │   app/pages/   │             │ app/components │
       │ - screening    │             │ - status_cards │
       │ - analysis     │             │ - fundus_viewer│
       │ - explain      │             │ - gradcam_view │
       │ - report       │             │ - metrics      │
       │ - dashboard    │             │ - report_comp  │
       └───────┬────────┘             └────────────────┘
               │
       ┌───────▼──────────────────────────┐
       │   app/utils/pipeline_adapter.py  │
       └───────┬───────────────────┬──────┘
               │                   │
  [Real Pipeline Present]  [Fallback / Testing]
               │                   │
       ┌───────▼────────┐  ┌───────▼────────┐
       │  src.pipeline  │  │  app/utils/    │
       │ analyze_fundus │  │  mock_pipeline │
       └────────────────┘  └────────────────┘
```

---

## 2. Page & Directory Structure

```
app/
├── main.py                     # Entrypoint & multi-page controller
├── pages/
│   ├── screening.py            # Patient registration & image upload
│   ├── analysis.py             # Quality assessment & anatomical landmarks
│   ├── explainability.py       # Grad-CAM attention & lesion evidence
│   ├── report.py               # Screening summary & ReportLab PDF export
│   └── dashboard.py            # District-level operational KPI dashboard
├── components/
│   ├── status_cards.py         # Severity, referral & gradability cards
│   ├── metrics.py              # KPI metric cards & summary rows
│   ├── fundus_viewer.py        # Retinal image & comparison viewer
│   ├── gradcam_viewer.py       # Explainability heatmap viewer & placeholders
│   ├── lesion_overlay.py       # Lesion counts & segmentation mask viewer
│   └── report_components.py    # Report layout & PDF generator
└── utils/
    ├── session.py              # Streamlit session-state management
    ├── formatting.py           # Severity maps, ICDR grading & color tags
    ├── mock_pipeline.py        # Isolated contract-compliant mock adapter
    └── pipeline_adapter.py     # Safe unified adapter with automatic fallback
```

---

## 3. Strict AI Integration Contract

The UI communicates with the AI screening pipeline exclusively via `analyze_fundus(image)`.

### Function Signature
```python
from app.utils.pipeline_adapter import analyze_fundus

result = analyze_fundus(image)
```

### Required Return Schema
```json
{
    "is_mock": false,
    "quality": {
        "score": 0.88,
        "gradable": true,
        "blur_score": 0.91,
        "illumination_score": 0.85,
        "field_of_view_score": 0.90,
        "feedback": ["Adequate optic disc and macular visualization"]
    },
    "preprocessing": {
        "enhanced_image": "<np.ndarray or None>"
    },
    "structures": {
        "vessels": "<np.ndarray or None>",
        "optic_disc": "<dict or None>",
        "fovea": "<dict or None>"
    },
    "lesions": {
        "microaneurysms": 0,
        "hemorrhages": 0,
        "hard_exudates": 0,
        "soft_exudates": 0
    },
    "grading": {
        "class_id": 2,
        "label": "Moderate Non-Proliferative DR",
        "referable": true,
        "confidence": 0.89,
        "probabilities": [0.03, 0.05, 0.89, 0.02, 0.01]
    },
    "explainability": {
        "gradcam": "<np.ndarray or None>",
        "lesion_evidence": []
    },
    "recommendation": "Referral to ophthalmology clinic recommended..."
}
```

---

## 4. Mock Mode & Safety Guarantees

1. **Explicit Identification:** When `src.pipeline` is not present, `pipeline_adapter.py` seamlessly falls back to `mock_pipeline.py`.
2. **Prominent Banner:** Every screen displays a `⚠️ DEMO / MOCK RESULT` badge whenever mock output is active.
3. **No Fabricated Evidence:** Mock mode never creates misleading Grad-CAM heatmaps or imaginary lesion counts.

---

## 5. How the Real AI Pipeline Connects

When the teammate finishes the AI model on their branch:
1. They export `analyze_fundus(image)` inside `src/pipeline.py`.
2. `pipeline_adapter.py` automatically detects `src.pipeline` on import.
3. The UI automatically transitions from mock mode to live AI inference without any code modifications required in the UI pages.

---

## 6. Running and Testing the Application

### 1. Activate Environment
```powershell
.venv\Scripts\activate
```

### 2. Run Test Suite
```powershell
pytest tests/test_ui_helpers.py
```

### 3. Launch Streamlit Application
```powershell
streamlit run app/main.py
```
