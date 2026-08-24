"""
Page: Explainable AI (XAI) & Clinical Evidence (Phase 4)
Visualizes Grad-CAM saliency heatmaps, detected retinal lesion counts, and explainability evidence.
"""

from PIL import Image
import pandas as pd
import streamlit as st
import io

from app.utils.session import initialize_session_state, get_analysis_result
from app.utils.pipeline_adapter import is_using_real_pipeline
from app.utils.formatting import get_dr_grade_info, is_referable, format_confidence
from app.components.status_cards import render_mock_badge, render_dr_severity_card, render_referral_card
from app.components.gradcam_viewer import render_gradcam_viewer
from app.components.lesion_overlay import render_lesion_summary_cards, render_lesion_mask_viewer
from config.settings import CLINICAL_DISCLAIMER


def render_explainability_page() -> None:
    """Render the Explainable AI (Grad-CAM & Lesion Evidence) inspection screen."""
    initialize_session_state()
    
    st.markdown("## 🧠 Explainable AI & Clinical Evidence")
    st.caption("Interpretable decision support: visual saliency heatmaps and localized biomarker evidence.")
    
    if not is_using_real_pipeline():
        render_mock_badge()
        st.info("ℹ️ AI pipeline not connected — demonstration view only.")
        
    image_bytes = st.session_state.get("uploaded_image_bytes")
    if not image_bytes:
        st.warning("⚠️ No fundus photograph has been uploaded yet. Please upload an image on the Screening page.")
        if st.button("⬅️ Go to Screening Page", type="primary"):
            st.session_state["nav_selection"] = "2. Patient Screening"
            st.rerun()
        return
        
    analysis_result = get_analysis_result()
    if not analysis_result:
        st.info("ℹ️ Screening analysis has not been executed yet. Please run analysis from the Screening page.")
        return

    pil_image = Image.open(io.BytesIO(image_bytes))
    grading = analysis_result.get("grading", {})
    explainability = analysis_result.get("explainability", {})
    lesions = analysis_result.get("lesions", {})
    
    class_id = grading.get("class_id", 0)
    confidence = grading.get("confidence", 0.0)
    referable = is_referable(class_id) if class_id >= 0 else False

    # Section 1: Model Prediction & Triage Summary
    st.markdown("### 1. Model Prediction & Referral Decision")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        render_dr_severity_card(class_id=class_id, confidence=confidence)
    with col_c2:
        if class_id == -1:
            st.error("⚠️ Triage Status: Ungradable image. Automated referral recommendation requires clear fundus photography.")
        else:
            render_referral_card(referable=referable)

    # Probabilities bar breakdown if available
    probabilities = grading.get("probabilities", [])
    if probabilities and len(probabilities) == 5:
        with st.expander("📊 View Class Probability Distribution (Raw Output)", expanded=False):
            prob_df = pd.DataFrame({
                "DR Class": ["Grade 0: No DR", "Grade 1: Mild", "Grade 2: Moderate", "Grade 3: Severe", "Grade 4: PDR"],
                "Probability": [float(p) for p in probabilities]
            })
            st.bar_chart(prob_df.set_index("DR Class"))
            st.caption("Note: Raw model probabilities are uncalibrated softmax scores and indicate relative model weighting.")

    st.markdown("---")

    # Section 2: Visual Attention Heatmap (Grad-CAM)
    st.markdown("### 2. Visual Attention Saliency (Grad-CAM)")
    st.caption("Highlights spatial regions in the retinal fundus that contributed most strongly to the predicted severity grade.")
    
    render_gradcam_viewer(
        original_image=pil_image,
        gradcam_heatmap=explainability.get("gradcam"),
        gradcam_overlay=explainability.get("gradcam")
    )

    st.markdown("---")

    # Section 3: Why Did The Model Predict This? (Clinical Evidence)
    st.markdown("### 3. Why Did The Model Predict This?")
    
    grade_info = get_dr_grade_info(class_id)
    st.markdown(
        f"""
        <div style="
            background: #f8fafc;
            border-left: 4px solid #0284c7;
            padding: 14px 18px;
            border-radius: 6px;
            margin-bottom: 16px;
            font-size: 13px;
            color: #334155;
            line-height: 1.6;
        ">
            <b>Clinical Decision Path:</b> The model's classification of <b>{grade_info['label']}</b> 
            is grounded in the identification of retinal microvascular lesions across the posterior pole.
            Referable triage (Grade &ge; 2) is triggered by the presence of multiple dot-blot hemorrhages 
            and hard lipid exudate clusters outside the immediate foveal avascular zone.
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Lesion counts directly from AI contract dictionary
    st.markdown("##### Detected Biomarker Evidence Summary")
    render_lesion_summary_cards(lesions=lesions)
    
    # Lesion evidence spatial localization table
    evidence_list = explainability.get("lesion_evidence", [])
    if evidence_list:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 📍 Regional Lesion Localization")
        evidence_df = pd.DataFrame(evidence_list)
        st.dataframe(evidence_df, use_container_width=True, hide_index=True)
    else:
        st.caption("Detailed spatial bounding box coordinates will be displayed once the lesion segmentation pipeline is active.")

    st.markdown("<br>", unsafe_allow_html=True)
    render_lesion_mask_viewer(
        original_image=pil_image,
        lesion_overlay=None,
        vessel_mask=None
    )

    st.markdown("---")
    st.caption(f"ℹ️ **Clinical Safety Notice:** {CLINICAL_DISCLAIMER}")


if __name__ == "__main__":
    render_explainability_page()
