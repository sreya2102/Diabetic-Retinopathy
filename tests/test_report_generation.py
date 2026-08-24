"""
Unit Tests for Phase 5: Clinical Screening Report & ReportLab PDF Generator
"""

import os
from app.components.report_components import generate_pdf_report, save_pdf_report_to_disk
from app.utils.mock_pipeline import analyze_fundus
from app.utils.image_processing import create_demo_fundus_sample
from PIL import Image
import io


def test_generate_pdf_report_valid_bytes():
    """Verify that ReportLab builds a valid PDF document matching the %PDF- header standard."""
    raw_bytes, _ = create_demo_fundus_sample("moderate_npdr")
    pil_img = Image.open(io.BytesIO(raw_bytes))
    analysis_result = analyze_fundus(pil_img)

    patient_data = {
        "patient_id": "PAT-TEST-001",
        "patient_age": 58,
        "patient_gender": "Female",
        "district": "Wayanad",
        "phc_center": "Meenangadi CHC"
    }

    pdf_bytes = generate_pdf_report(
        patient_data=patient_data,
        analysis_result=analysis_result,
        image_bytes=raw_bytes,
        timestamp="2026-08-24 20:00:00"
    )

    assert pdf_bytes is not None
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF-")


def test_save_pdf_report_to_disk():
    """Verify that clinical reports can be archived to reports/generated/."""
    dummy_pdf_bytes = b"%PDF-1.4 dummy clinical report data"
    saved_path = save_pdf_report_to_disk(dummy_pdf_bytes, "PAT-TEST-002")

    assert os.path.exists(saved_path)
    assert saved_path.startswith(os.path.join("reports", "generated"))

    # Cleanup test artifact
    try:
        os.remove(saved_path)
    except OSError:
        pass
