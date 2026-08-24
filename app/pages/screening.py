import os
import sys
from pathlib import Path

# Add project root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datetime import datetime
from PIL import Image
import streamlit as st
import io

from app.utils.session import (
    initialize_session_state,
    set_patient_data,
    set_uploaded_image,
    set_analysis_result,
    get_patient_data,
    get_analysis_result
)
from app.utils.image_processing import (
    validate_image_file,
    estimate_image_quality,
    create_demo_fundus_sample
)
from app.utils.pipeline_adapter import analyze_fundus, is_using_real_pipeline
from app.components.status_cards import render_mock_badge, render_quality_card, render_referral_card, render_dr_severity_card
from app.components.fundus_viewer import render_fundus_viewer
from app.components.metrics import render_metrics_row
from config.settings import SUPPORTED_IMAGE_TYPES, CLINICAL_DISCLAIMER


def render_screening_page() -> None:
    """Render the complete patient intake, image acquisition, and screening triage workflow."""
    initialize_session_state()
    
    st.markdown("## 📋 Patient Registration & Fundus Acquisition")
    st.caption("Register patient demographics and acquire retinal fundus photography for tele-ophthalmology screening.")
    
    if not is_using_real_pipeline():
        render_mock_badge()

    # Step 1: Patient Demographics Form
    st.markdown("### 1. Patient Information")
    existing_patient = get_patient_data()
    
    with st.container():
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            patient_id = st.text_input(
                "Patient ID / National Health ID *",
                value=existing_patient.get("patient_id") or "PAT-2026-0842",
                placeholder="e.g. PAT-2026-0842"
            )
        with col2:
            patient_age = st.number_input(
                "Age (Years)",
                min_value=1,
                max_value=120,
                value=existing_patient.get("patient_age") or 54,
                step=1
            )
        with col3:
            gender_options = ["Female", "Male", "Other", "Unspecified"]
            curr_gender = existing_patient.get("patient_gender", "Female")
            gender_idx = gender_options.index(curr_gender) if curr_gender in gender_options else 0
            patient_gender = st.selectbox("Gender", options=gender_options, index=gender_idx)

        col4, col5 = st.columns(2)
        with col4:
            district = st.text_input(
                "District / Region",
                value=existing_patient.get("district") or "Wayanad District, Kerala",
                placeholder="e.g. Wayanad, Kerala"
            )
        with col5:
            phc_center = st.text_input(
                "Primary Health Centre (PHC)",
                value=existing_patient.get("phc_center") or "Meenangadi Community Health Centre",
                placeholder="e.g. Meenangadi CHC"
            )

    st.markdown("---")
    
    # Step 2: Fundus Image Acquisition
    st.markdown("### 2. Fundus Image Acquisition")
    st.caption("Acquire a 45° posterior pole fundus photograph via direct file upload or clinic testing library.")
    
    acquisition_tab1, acquisition_tab2 = st.tabs([
        "📁 Upload Fundus Image",
        "🏥 Custom File / Clinic Demo Library"
    ])
    
    selected_image_bytes = None
    selected_filename = None
    
    with acquisition_tab1:
        uploaded_file = st.file_uploader(
            "Select retinal fundus photograph (JPG, JPEG, PNG)",
            type=SUPPORTED_IMAGE_TYPES,
            help="High-resolution, centered 45° or 50° field-of-view fundus image.",
            key="fundus_file_uploader"
        )
        if uploaded_file is not None:
            raw_bytes = uploaded_file.getvalue()
            is_valid, err_msg, validated_img = validate_image_file(raw_bytes, uploaded_file.name)
            if not is_valid:
                st.error(f"❌ Validation Error: {err_msg}")
            else:
                selected_image_bytes = raw_bytes
                selected_filename = uploaded_file.name

    with acquisition_tab2:
        st.markdown("**Select a standardized clinical test case for system testing:**")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        
        with col_s1:
            if st.button("🟢 Normal Fundus\n(Grade 0)", use_container_width=True):
                selected_image_bytes, selected_filename = create_demo_fundus_sample("normal")
        with col_s2:
            if st.button("🟡 Moderate NPDR\n(Referable)", use_container_width=True):
                selected_image_bytes, selected_filename = create_demo_fundus_sample("moderate_npdr")
        with col_s3:
            if st.button("🔴 Ungradable (Blur)\n(Defocus)", use_container_width=True):
                selected_image_bytes, selected_filename = create_demo_fundus_sample("ungradable_blur")
        with col_s4:
            if st.button("🔴 Ungradable (Glare)\n(Overexposed)", use_container_width=True):
                selected_image_bytes, selected_filename = create_demo_fundus_sample("ungradable_glare")

    # If an image has been acquired either in this step or previously in session:
    if selected_image_bytes is not None:
        try:
            pil_img = Image.open(io.BytesIO(selected_image_bytes))
            size_kb = len(selected_image_bytes) / 1024.0
            set_uploaded_image(
                image_bytes=selected_image_bytes,
                filename=selected_filename,
                image_format=pil_img.format or "JPEG",
                size_kb=size_kb
            )
        except Exception as e:
            st.error(f"Error reading image: {str(e)}")
            
    active_bytes = st.session_state.get("uploaded_image_bytes")
    active_filename = st.session_state.get("uploaded_image_filename", "fundus_capture.jpg")
    
    if active_bytes:
        pil_img = Image.open(io.BytesIO(active_bytes))
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_preview, col_qual = st.columns([1.2, 1.8])
        
        with col_preview:
            st.markdown("##### 👁️ Acquired Retinal Fundus")
            render_fundus_viewer(
                pil_img,
                caption=f"Source: {active_filename}",
                metadata={
                    "Resolution": f"{pil_img.width} × {pil_img.height} px",
                    "Format": pil_img.format or "JPEG",
                    "Size": f"{len(active_bytes)/1024.0:.1f} KB"
                }
            )
            
        with col_qual:
            st.markdown("##### 🔍 Automated Image Quality Inspection")
            # Dynamic quality estimation
            quality_data = estimate_image_quality(pil_img)
            is_gradable = quality_data.get("gradable", True)
            
            render_metrics_row([
                {
                    "label": "Quality Score",
                    "value": f"{quality_data.get('score', 0.0)*100:.0f}%",
                    "subtext": "Optical Clarity Index",
                    "border_color": "#10b981" if is_gradable else "#ef4444"
                },
                {
                    "label": "Sharpness / Focus",
                    "value": f"{quality_data.get('blur_score', 0.0)*100:.0f}%",
                    "subtext": "Vessel edge contrast",
                },
                {
                    "label": "Illumination",
                    "value": f"{quality_data.get('illumination_score', 0.0)*100:.0f}%",
                    "subtext": "Luminance balance",
                }
            ])
            
            render_quality_card(
                gradable=is_gradable,
                score=quality_data.get("score", 0.0),
                feedback=quality_data.get("feedback")
            )
            
            # UNGRADABLE RECAPTURE PROTOCOL
            if not is_gradable:
                st.markdown(
                    """
                    <div style="
                        background-color: #fef2f2;
                        border: 2px solid #ef4444;
                        border-radius: 8px;
                        padding: 16px;
                        margin-top: 12px;
                        color: #991b1b;
                    ">
                        <div style="font-size: 15px; font-weight: 700; display: flex; align-items: center;">
                            ⚠️ Image May Be Ungradable — Recapture Required
                        </div>
                        <div style="font-size: 13px; margin-top: 6px; line-height: 1.5; color: #7f1d1d;">
                            <b>Suggested action:</b> Recapture retinal photograph with improved focus, 
                            reduced glare, and adequate pupil alignment.
                        </div>
                        <div style="font-size: 12px; margin-top: 6px; color: #b91c1c;">
                            <i>Safety Protocol: Automated DR diagnosis is disabled on ungradable imagery to prevent false reassurance or erroneous referrals.</i>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.success("✅ Fundus photograph meets clinical gradability criteria for diagnostic screening.")
                
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Run AI Screening Analysis", type="primary", use_container_width=True):
                # Save patient data
                set_patient_data(
                    patient_id=patient_id,
                    age=int(patient_age) if patient_age else None,
                    gender=patient_gender,
                    district=district,
                    phc_center=phc_center
                )
                st.session_state["screening_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                with st.spinner("Processing retinal vascular landmarks, lesion segmentation, and DR grading..."):
                    analysis_output = analyze_fundus(pil_img)
                    set_analysis_result(analysis_output)
                    
                st.success("Screening analysis completed successfully! Explore detailed findings via the navigation menu.")
                st.rerun()

    # If an analysis result is already available in the session, display a summary preview
    cached_result = get_analysis_result()
    if cached_result:
        st.markdown("---")
        st.markdown("### 3. Immediate Screening Result Snapshot")
        grading = cached_result.get("grading", {})
        class_id = grading.get("class_id", 0)
        
        if class_id == -1:
            st.error("⚠️ Screening Outcome: UNGRADABLE IMAGE. Please recapture the fundus image.")
        else:
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                render_dr_severity_card(class_id=class_id, confidence=grading.get("confidence"))
            with col_res2:
                render_referral_card(referable=bool(grading.get("referable", False)))
                
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔍 View Anatomical & Quality Analysis", use_container_width=True):
                st.session_state["nav_selection"] = "3. Quality & Structures"
                st.rerun()
        with col_btn2:
            if st.button("🧠 View Explainable AI & Grad-CAM Evidence", use_container_width=True):
                st.session_state["nav_selection"] = "4. Explainable AI (Grad-CAM)"
                st.rerun()

    st.markdown("---")
    st.caption(f"ℹ️ **Clinical Safety Notice:** {CLINICAL_DISCLAIMER}")


if __name__ == "__main__":
    render_screening_page()
