"""
Clinical Report Components & ReportLab PDF Generator (Phase 5)
Renders professional on-screen screening summary blocks and exports multi-section clinical PDF reports.
"""

from datetime import datetime
import io
import os
from typing import Any, Dict, Optional
from PIL import Image as PILImage
import streamlit as st

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from app.utils.formatting import get_dr_grade_info, is_referable, format_confidence
from config.settings import CLINICAL_DISCLAIMER, APP_NAME, APP_SUBTITLE


def render_report_header(patient_data: Dict[str, Any], timestamp: Optional[str] = None) -> None:
    """Render the official header and patient demographic table for the screening report."""
    report_time = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    st.markdown(
        f"""
        <div style="border-bottom: 2px solid #0284c7; padding-bottom: 12px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: flex-end;">
                <div>
                    <h2 style="margin: 0; color: #0f172a; font-size: 24px; font-weight: 700;">👁️ {APP_NAME}</h2>
                    <div style="font-size: 13px; color: #0284c7; font-weight: 600;">{APP_SUBTITLE}</div>
                </div>
                <div style="text-align: right; font-size: 12px; color: #64748b;">
                    <div><b>Report Generated:</b> {report_time}</div>
                    <div><b>Protocol:</b> Tele-Screening Protocol v1.0</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Patient Demographic Overview Table
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"**Patient ID:** `{patient_data.get('patient_id') or 'N/A'}`")
    with col2:
        age_val = patient_data.get('patient_age')
        age_str = f"{age_val} yrs" if age_val else "N/A"
        st.markdown(f"**Age / Gender:** {age_str} ({patient_data.get('patient_gender', 'Unspecified')})")
    with col3:
        st.markdown(f"**District:** {patient_data.get('district') or 'Rural District'}")
    with col4:
        st.markdown(f"**PHC Center:** {patient_data.get('phc_center') or 'Primary Health Centre'}")


def render_disclaimer_block() -> None:
    """Render mandatory clinical disclaimer and AI limitation notice."""
    st.markdown(
        f"""
        <div style="
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-left: 4px solid #64748b;
            border-radius: 6px;
            padding: 14px 18px;
            margin-top: 24px;
            font-size: 12px;
            color: #475569;
            line-height: 1.5;
        ">
            <b>Clinical & Regulatory Notice:</b> {CLINICAL_DISCLAIMER}
            <div style="margin-top: 4px; font-size: 11px; color: #64748b;">
                AI-assisted screening output. This system does not replace examination or diagnosis by a qualified ophthalmologist.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def generate_pdf_report(
    patient_data: Dict[str, Any],
    analysis_result: Dict[str, Any],
    image_bytes: Optional[bytes] = None,
    timestamp: Optional[str] = None
) -> bytes:
    """
    Generate a complete, publication-grade clinical PDF screening report using ReportLab.
    Includes patient demographics, embedded fundus thumbnail, findings table, lesion counts,
    actionable recommendation, ophthalmologist review block, and disclaimer.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a")
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#0284c7")
    )
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=8,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155")
    )
    disclaimer_style = ParagraphStyle(
        'DisclaimerCustom',
        parent=styles['Normal'],
        fontSize=7.5,
        leading=10.5,
        textColor=colors.HexColor("#64748b")
    )

    story = []
    
    # 1. Header Banner
    story.append(Paragraph(f"<b>{APP_NAME}</b> &bull; DIABETIC RETINOPATHY SCREENING REPORT", title_style))
    story.append(Paragraph(APP_SUBTITLE, subtitle_style))
    story.append(Spacer(1, 8))
    
    # Demo banner if mock
    if analysis_result.get("is_mock", True):
        demo_style = ParagraphStyle(
            'DemoWarning',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor("#b45309")
        )
        story.append(Paragraph("<b>[DEMO / MOCK REPORT — For Testing and System Demonstration Only]</b>", demo_style))
        story.append(Spacer(1, 4))

    # 2. Patient Demographic Table
    report_time = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    age_val = patient_data.get('patient_age')
    age_str = f"{age_val} yrs" if age_val else "N/A"
    
    patient_table_data = [
        [
            Paragraph(f"<b>Patient ID:</b> {patient_data.get('patient_id', 'N/A')}", body_style),
            Paragraph(f"<b>Age / Gender:</b> {age_str} / {patient_data.get('patient_gender', 'N/A')}", body_style)
        ],
        [
            Paragraph(f"<b>District:</b> {patient_data.get('district', 'N/A')}", body_style),
            Paragraph(f"<b>PHC Centre:</b> {patient_data.get('phc_center', 'N/A')}", body_style)
        ],
        [
            Paragraph(f"<b>Acquisition Date:</b> {report_time}", body_style),
            Paragraph("<b>Screening Modality:</b> 45° Non-Mydriatic Fundus", body_style)
        ]
    ]
    t_patient = Table(patient_table_data, colWidths=[270, 270])
    t_patient.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_patient)
    story.append(Spacer(1, 8))

    # 3. Diagnostic Assessment & Findings Table
    grading = analysis_result.get("grading", {})
    quality = analysis_result.get("quality", {})
    class_id = grading.get("class_id", 0)
    grade_info = get_dr_grade_info(class_id)
    referable = is_referable(class_id) if class_id >= 0 else False
    
    story.append(Paragraph("<b>1. AI Screening Assessment & Diagnostic Triage</b>", section_heading))
    
    ref_label = "<b>REFERRAL REQUIRED (Referable DR)</b>" if referable else "Non-Referable (Routine Follow-up)"
    if class_id == -1:
        ref_label = "<b>RECAPTURE REQUIRED (Ungradable Image)</b>"

    findings_data = [
        [Paragraph("<b>Clinical Metric</b>", body_style), Paragraph("<b>Screening Finding</b>", body_style)],
        [Paragraph("DR Severity Grade", body_style), Paragraph(f"<b>Grade {class_id}: {grade_info['label']}</b>", body_style)],
        [Paragraph("Tele-Screening Decision", body_style), Paragraph(ref_label, body_style)],
        [Paragraph("Model Confidence", body_style), Paragraph(format_confidence(grading.get("confidence")), body_style)],
        [Paragraph("Image Gradability", body_style), Paragraph("Gradable" if quality.get("gradable", True) else "Ungradable (Recapture Advised)", body_style)],
        [Paragraph("Quality Score", body_style), Paragraph(f"{quality.get('score', 0.0)*100:.0f}% (Focus: {quality.get('blur_score', 0.0)*100:.0f}%, Illumination: {quality.get('illumination_score', 0.0)*100:.0f}%)", body_style)],
    ]
    t_findings = Table(findings_data, colWidths=[180, 360])
    t_findings.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_findings)
    story.append(Spacer(1, 8))

    # 4. Lesion Biomarker Evidence
    lesions = analysis_result.get("lesions", {})
    story.append(Paragraph("<b>2. Lesion Evidence Breakdown</b>", section_heading))
    lesion_table_data = [
        [
            Paragraph(f"<b>Microaneurysms:</b> {lesions.get('microaneurysms', 0)}", body_style),
            Paragraph(f"<b>Hemorrhages:</b> {lesions.get('hemorrhages', 0)}", body_style),
            Paragraph(f"<b>Hard Exudates:</b> {lesions.get('hard_exudates', 0)}", body_style),
            Paragraph(f"<b>Soft Exudates:</b> {lesions.get('soft_exudates', 0)}", body_style),
        ]
    ]
    t_lesions = Table(lesion_table_data, colWidths=[135, 135, 135, 135])
    t_lesions.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_lesions)
    story.append(Spacer(1, 8))

    # 5. Actionable Recommendation
    rec = analysis_result.get("recommendation", grade_info["referral_recommendation"])
    story.append(Paragraph("<b>3. Actionable Clinical Recommendation</b>", section_heading))
    story.append(Paragraph(rec, body_style))
    story.append(Spacer(1, 10))

    # 6. Tele-Ophthalmologist Review & Validation Block
    story.append(Paragraph("<b>4. Tele-Ophthalmologist Clinical Review</b>", section_heading))
    doc_review_data = [
        [
            Paragraph("<b>Reviewing Ophthalmologist:</b> ___________________________", body_style),
            Paragraph("<b>Medical Registration No:</b> _______________", body_style)
        ],
        [
            Paragraph("<b>Clinical Validation:</b> [  ] Approved AI Grade   [  ] Modified Grade: ____", body_style),
            Paragraph("<b>Signature:</b> _________________________", body_style)
        ]
    ]
    t_doc = Table(doc_review_data, colWidths=[310, 230])
    t_doc.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_doc)
    story.append(Spacer(1, 10))

    # 7. Regulatory Disclaimer
    story.append(Paragraph(
        f"<b>Regulatory & Legal Disclaimer:</b> {CLINICAL_DISCLAIMER} "
        "AI-assisted screening output. This system does not replace examination or diagnosis by a qualified ophthalmologist.",
        disclaimer_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def save_pdf_report_to_disk(pdf_bytes: bytes, patient_id: str) -> str:
    """Save the generated clinical PDF report to reports/generated/ and return path."""
    os.makedirs("reports/generated", exist_ok=True)
    slug = patient_id.replace(" ", "_").replace("/", "-") if patient_id else "patient"
    filename = f"RETINASCAN_Report_{slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = os.path.join("reports", "generated", filename)
    with open(filepath, "wb") as f:
        f.write(pdf_bytes)
    return filepath
