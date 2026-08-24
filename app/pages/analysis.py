import os
import sys
from pathlib import Path

# Add project root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from PIL import Image
import streamlit as st
import io

from app.utils.session import initialize_session_state, get_analysis_result
from app.utils.pipeline_adapter import is_using_real_pipeline
from app.utils.image_processing import (
    apply_clahe_enhancement,
    extract_vessel_mask,
    locate_retinal_landmarks,
    generate_annotated_fundus_overlay
)
from app.components.status_cards import render_mock_badge, render_quality_card
from app.components.fundus_viewer import render_side_by_side_comparison, render_fundus_viewer
from app.components.metrics import render_metrics_row
from config.settings import CLINICAL_DISCLAIMER


def render_analysis_page() -> None:
    """Render the comprehensive image quality, enhancement, and anatomical structures analysis page."""
    initialize_session_state()
    
    st.markdown("## 🔬 Retinal Image Quality & Structure Analysis")
    st.caption("Detailed assessment of optical quality, microvascular enhancement, and key retinal landmarks.")
    
    if not is_using_real_pipeline():
        render_mock_badge()
        
    image_bytes = st.session_state.get("uploaded_image_bytes")
    if not image_bytes:
        st.warning("⚠️ No fundus photograph has been uploaded yet. Please proceed to the Screening page first.")
        if st.button("⬅️ Go to Screening Page", type="primary"):
            st.session_state["nav_selection"] = "2. Patient Screening"
            st.rerun()
        return
        
    analysis_result = get_analysis_result()
    if not analysis_result:
        st.info("ℹ️ Image uploaded, but screening analysis has not been executed yet. Click 'Run Screening Analysis' on the Screening page.")
        return

    pil_image = Image.open(io.BytesIO(image_bytes))
    quality = analysis_result.get("quality", {})
    
    # Compute CLAHE enhanced image dynamically if not supplied by pipeline
    enhanced_img = analysis_result.get("preprocessing", {}).get("enhanced_image")
    if enhanced_img is None:
        enhanced_img = apply_clahe_enhancement(pil_image)

    # Section 1: Quality Breakdown Metrics
    st.markdown("### 1. Multi-Parameter Image Quality Index")
    is_gradable = quality.get("gradable", True)
    
    render_metrics_row([
        {
            "label": "Overall Quality Score",
            "value": f"{quality.get('score', 0.0) * 100:.0f}%",
            "subtext": "Gradable" if is_gradable else "Ungradable",
            "border_color": "#10b981" if is_gradable else "#ef4444"
        },
        {
            "label": "Focus / Sharpness",
            "value": f"{quality.get('blur_score', 0.0) * 100:.0f}%",
            "subtext": "Laplacian variance measure",
        },
        {
            "label": "Illumination Uniformity",
            "value": f"{quality.get('illumination_score', 0.0) * 100:.0f}%",
            "subtext": "Luminance distribution",
        },
        {
            "label": "Field of View",
            "value": f"{quality.get('field_of_view_score', 0.0) * 100:.0f}%",
            "subtext": "45° Posterior pole coverage",
        }
    ])
    
    render_quality_card(
        gradable=is_gradable,
        score=quality.get("score", 0.0),
        feedback=quality.get("feedback")
    )
    
    st.markdown("---")

    # Section 2: Visual Comparison & Interactive Overlays
    st.markdown("### 2. Preprocessing & Structure Visualizations")
    
    view_mode = st.radio(
        "Select Inspection View:",
        options=[
            "Side-by-Side: Raw vs CLAHE Enhanced",
            "Interactive Anatomical Overlays",
            "Retinal Vasculature Tree (Vessel Mask)"
        ],
        horizontal=True
    )
    
    if view_mode == "Side-by-Side: Raw vs CLAHE Enhanced":
        render_side_by_side_comparison(
            left_image=pil_image,
            right_image=enhanced_img,
            left_title="Original Raw Fundus",
            right_title="Green-Channel CLAHE Enhanced",
            right_placeholder_text="Contrast enhancement highlights fine microvascular details."
        )
    elif view_mode == "Interactive Anatomical Overlays":
        col_ctrl, col_annot = st.columns([1, 3])
        with col_ctrl:
            st.markdown("##### Layer Controls")
            show_vessels = st.checkbox("Retinal Blood Vessels", value=True)
            show_disc = st.checkbox("Optic Disc (OD)", value=True)
            show_fovea = st.checkbox("Fovea / Macula", value=True)
            
            st.caption("Toggle specific retinal anatomical layers to inspect landmark boundaries and vessel clarity.")
        with col_annot:
            annotated_img = generate_annotated_fundus_overlay(
                image=pil_image,
                show_vessels=show_vessels,
                show_disc=show_disc,
                show_fovea=show_fovea
            )
            render_fundus_viewer(
                annotated_img,
                caption="Anatomical Landmarks & Vascular Overlay"
            )
    else:  # Retinal Vasculature Tree
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.markdown("##### Raw Fundus")
            render_fundus_viewer(pil_image, caption="Base Photograph")
        with col_v2:
            st.markdown("##### Extracted Vascular Tree")
            vessel_img = extract_vessel_mask(pil_image)
            render_fundus_viewer(vessel_img, caption="Morphological Vessel Segmentation")

    st.markdown("---")

    # Section 3: Key Anatomical Landmarks Inspection
    st.markdown("### 3. Anatomical Landmarks & Clinical Reference")
    landmarks = locate_retinal_landmarks(pil_image)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        od = landmarks["optic_disc"]
        st.markdown(
            f"""
            <div style="border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; background: white;">
                <div style="font-size: 13px; font-weight: 700; color: #d97706;">🟡 OPTIC DISC (OD)</div>
                <div style="font-size: 18px; font-weight: 700; color: #0f172a; margin: 4px 0;">{od['margin_status']}</div>
                <div style="font-size: 12px; color: #64748b;">
                    <b>Coordinates:</b> {od['center']} &bull; <b>Radius:</b> {od['radius']}px<br>
                    <b>Estimated C/D Ratio:</b> {od['cup_disc_ratio_est']:.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        fov = landmarks["fovea"]
        st.markdown(
            f"""
            <div style="border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; background: white;">
                <div style="font-size: 13px; font-weight: 700; color: #059669;">🟢 FOVEA / MACULA</div>
                <div style="font-size: 18px; font-weight: 700; color: #0f172a; margin: 4px 0;">{fov['avascular_zone_status']}</div>
                <div style="font-size: 12px; color: #64748b;">
                    <b>Coordinates:</b> {fov['center']} &bull; <b>Radius:</b> {fov['radius']}px<br>
                    <b>Foveal Threat:</b> {fov['exudate_proximity']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        vasc = landmarks["vasculature"]
        st.markdown(
            f"""
            <div style="border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; background: white;">
                <div style="font-size: 13px; font-weight: 700; color: #0284c7;">🔵 RETINAL VASCULATURE</div>
                <div style="font-size: 18px; font-weight: 700; color: #0f172a; margin: 4px 0;">{vasc['arcade_status']}</div>
                <div style="font-size: 12px; color: #64748b;">
                    <b>Vessel Density Index:</b> {vasc['vessel_density_index']:.2f}<br>
                    <b>Major Arcades:</b> Superior & Inferior Temporal
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.caption(f"ℹ️ **Clinical Safety Notice:** {CLINICAL_DISCLAIMER}")


if __name__ == "__main__":
    render_analysis_page()
