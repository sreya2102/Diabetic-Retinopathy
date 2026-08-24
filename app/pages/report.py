"""
Page: Clinical Screening Report & PDF Export (Phase 5)
Presents an official clinical screening summary and allows exporting structured PDF reports.
"""

from datetime import datetime
import streamlit as st

from app.utils.session import (
    initialize_session_state,
    get_patient_data,
    get_analysis_result
)
from app.utils.pipeline_adapter import is_using_real_pipeline
from app.utils.formatting import get_dr_grade_info, is_referable, format_confidence
from app.components.status_cards import render_mock_badge
from app.components.report_components import (
    render_report_header,
    render_disclaimer_block,
    generate_pdf_report,
    save_pdf_report_to_disk
)
from app.components.lesion_overlay import render_lesion_summary_cards
from config.settings import CLINICAL_DISCLAIMER


def render_report_page() -> None:
    """Render the clinical screening report page with PDF export and archiving functionality."""
    initialize_session_state()
    
    st.markdown("## 📄 Clinical Screening Report & PDF Export")
    st.caption("Consolidated screening summary for tele-ophthalmologist review, EMR archival, and patient consultation.")
    
    if not is_using_real_pipeline():
        render_mock_badge()
        
    analysis_result = get_analysis_result()
    if not analysis_result:
        st.warning("⚠️ No screening analysis result found. Please upload a fundus photograph and run analysis on the Screening page.")
        if st.button("⬅️ Go to Screening Page", type="primary"):
            st.session_state["nav_selection"] = "2. Patient Screening"
            st.rerun()
        return

    patient_data = get_patient_data()
    image_bytes = st.session_state.get("uploaded_image_bytes")
    timestamp = st.session_state.get("screening_timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. Report Header & Patient Information
    render_report_header(patient_data=patient_data, timestamp=timestamp)
    
    # 2. Findings Summary
    grading = analysis_result.get("grading", {})
    quality = analysis_result.get("quality", {})
    lesions = analysis_result.get("lesions", {})
    class_id = grading.get("class_id", 0)
    grade_info = get_dr_grade_info(class_id)
    referable = is_referable(class_id) if class_id >= 0 else False
    
    st.markdown("### 1. Diagnostic Summary & Referral Triage")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div style="background: white; border: 1px solid #e2e8f0; border-top: 4px solid {grade_info['color']}; border-radius: 8px; padding: 14px;">
                <div style="font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase;">DIAGNOSTIC GRADE</div>
                <div style="font-size: 20px; font-weight: 700; color: {grade_info['color']}; margin-top: 2px;">
                    {grade_info['label']}
                </div>
                <div style="font-size: 12px; color: #64748b; margin-top: 2px;">Grade {class_id} on ICDR scale</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        ref_color = "#dc2626" if referable else "#059669"
        ref_text = "REFERRAL REQUIRED" if referable else "NON-REFERABLE"
        if class_id == -1:
            ref_color = "#ea580c"
            ref_text = "RECAPTURE REQUIRED"
            
        st.markdown(
            f"""
            <div style="background: white; border: 1px solid #e2e8f0; border-top: 4px solid {ref_color}; border-radius: 8px; padding: 14px;">
                <div style="font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase;">TELEMEDICINE TRIAGE</div>
                <div style="font-size: 20px; font-weight: 700; color: {ref_color}; margin-top: 2px;">
                    {ref_text}
                </div>
                <div style="font-size: 12px; color: #64748b; margin-top: 2px;">Triage Threshold: Grade &ge; 2</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""
            <div style="background: white; border: 1px solid #e2e8f0; border-top: 4px solid #0284c7; border-radius: 8px; padding: 14px;">
                <div style="font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase;">MODEL CONFIDENCE</div>
                <div style="font-size: 20px; font-weight: 700; color: #0f172a; margin-top: 2px;">
                    {format_confidence(grading.get('confidence'))}
                </div>
                <div style="font-size: 12px; color: #64748b; margin-top: 2px;">Optical Quality: {quality.get('score', 0.0)*100:.0f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # 3. Lesion Findings
    st.markdown("### 2. Clinical Lesion Evidence")
    render_lesion_summary_cards(lesions=lesions)

    st.markdown("---")

    # 4. Actionable Referral Recommendation
    st.markdown("### 3. Recommended Clinical Action")
    recommendation_text = analysis_result.get("recommendation", grade_info["referral_recommendation"])
    st.info(f"📋 **Action Plan:** {recommendation_text}")

    # 5. Mandatory Disclaimer Block
    render_disclaimer_block()

    st.markdown("---")

    # 6. PDF Export & Archive Section
    st.markdown("### 4. Export & Tele-Consultation")
    col_pdf, col_save, col_info = st.columns([1.5, 1.5, 2])
    
    pdf_bytes = generate_pdf_report(
        patient_data=patient_data,
        analysis_result=analysis_result,
        image_bytes=image_bytes,
        timestamp=timestamp
    )
    
    patient_slug = patient_data.get("patient_id") or "unnamed_patient"
    safe_filename = f"RETINASCAN_Report_{patient_slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    with col_pdf:
        st.download_button(
            label="📥 Download Clinical PDF Report",
            data=pdf_bytes,
            file_name=safe_filename,
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )
        
    with col_save:
        if st.button("💾 Save to District EMR Archive", use_container_width=True):
            saved_path = save_pdf_report_to_disk(pdf_bytes, patient_slug)
            st.success(f"Report archived to: `{saved_path}`")
            
    with col_info:
        st.caption("PDF reports are structured according to national tele-ophthalmology standards with ophthalmologist signature blocks.")


if __name__ == "__main__":
    render_report_page()
