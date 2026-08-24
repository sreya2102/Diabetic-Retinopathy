"""
RETINASCAN: Explainable AI for Rural Diabetic Retinopathy Screening
Main Application Entrypoint & Medical Screening Hub
"""

import streamlit as st

from app.utils.session import initialize_session_state
from app.utils.pipeline_adapter import is_using_real_pipeline
from app.components.status_cards import render_mock_badge
from app.components.metrics import render_metrics_row
from config.settings import APP_NAME, APP_SUBTITLE, CLINICAL_DISCLAIMER

# Import page modules
from app.pages.screening import render_screening_page
from app.pages.analysis import render_analysis_page
from app.pages.explainability import render_explainability_page
from app.pages.report import render_report_page
from app.pages.dashboard import render_dashboard_page
from app.pages.simulation import render_simulation_page

# Page Configuration
st.set_page_config(
    page_title=f"{APP_NAME} | Clinical Retinal Screening",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Clean Medical Design CSS
st.markdown(
    """
    <style>
        /* Base typography & clean palette */
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #1e293b;
        }
        
        /* Main background */
        .stApp {
            background-color: #f8fafc;
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #0f172a;
            color: #f8fafc;
        }
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label {
            color: #f8fafc !important;
        }
        
        /* Top Hero Banner */
        .retina-hero {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
            color: white;
            border-radius: 12px;
            padding: 32px 36px;
            margin-bottom: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .retina-hero h1 {
            color: #ffffff;
            font-size: 32px;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.5px;
        }
        .retina-hero p {
            color: #93c5fd;
            font-size: 16px;
            margin: 8px 0 0 0;
            font-weight: 500;
        }
        
        /* Workflow step cards */
        .step-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
            height: 100%;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }
        .step-icon {
            font-size: 22px;
            margin-bottom: 6px;
        }
        .step-title {
            font-weight: 700;
            font-size: 13px;
            color: #0f172a;
            margin-bottom: 4px;
        }
        .step-desc {
            font-size: 11px;
            color: #64748b;
            line-height: 1.4;
        }

        /* Buttons */
        .stButton>button {
            border-radius: 6px;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True
)

def render_home_page() -> None:
    """Render the main RETINASCAN landing overview page."""
    initialize_session_state()
    
    # Hero Section
    st.markdown(
        f"""
        <div class="retina-hero">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <h1>👁️ {APP_NAME}</h1>
                    <p>{APP_SUBTITLE}</p>
                </div>
                <div style="text-align: right;">
                    <span style="background: rgba(255,255,255,0.15); padding: 6px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; color: #e2e8f0; border: 1px solid rgba(255,255,255,0.2);">
                        Clinical Protocol v1.0
                    </span>
                </div>
            </div>
            <div style="margin-top: 16px; font-size: 14px; color: #cbd5e1; max-width: 850px; line-height: 1.6;">
                AI-assisted retinal screening designed to support early detection and ophthalmologist referral 
                in rural Primary Health Centres (PHCs) across India with low-bandwidth tele-connectivity.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if not is_using_real_pipeline():
        render_mock_badge()

    # Quick Navigation Call to Actions
    col_act1, col_act2, col_act3 = st.columns([1.5, 1.5, 1])
    with col_act1:
        if st.button("🚀 Start New Patient Screening", type="primary", use_container_width=True):
            st.session_state["nav_selection"] = "2. Patient Screening"
            st.rerun()
    with col_act2:
        if st.button("📊 Open District Dashboard", use_container_width=True):
            st.session_state["nav_selection"] = "6. District Dashboard"
            st.rerun()
    with col_act3:
        st.markdown(
            """
            <div style="text-align: center; padding-top: 6px; font-size: 12px; color: #64748b;">
                Triage Threshold: <b>Grade ≥ 2</b>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Overview Metrics Row
    st.markdown("### 📊 Screening Program At A Glance")
    render_metrics_row([
        {
            "label": "Rural Target Population",
            "value": "100,000+",
            "subtext": "Annual district screening capacity target",
        },
        {
            "label": "Triage Sensitivity Target",
            "value": "95.0%",
            "subtext": "For referable DR (Grade ≥ 2)",
            "border_color": "#10b981"
        },
        {
            "label": "Referral Turnaround",
            "value": "< 48 hrs",
            "subtext": "PHC to tele-ophthalmology triage",
            "border_color": "#0284c7"
        },
        {
            "label": "Bandwidth Optimization",
            "value": "Edge-Ready",
            "subtext": "Offline quality triage & compressed sync",
        }
    ])

    st.markdown("---")

    # Screening Workflow Simulation Overview
    st.markdown("### 🔄 Tele-Screening Pipeline Architecture")
    st.caption("Standardized clinical screening progression from rural community acquisition to tertiary specialist review.")
    
    col_w1, col_w2, col_w3, col_w4 = st.columns(4)
    with col_w1:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-icon">📸</div>
                <div class="step-title">1. Fundus Acquisition</div>
                <div class="step-desc">45° non-mydriatic fundus photograph captured at village PHC.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_w2:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-icon">🔍</div>
                <div class="step-title">2. Quality & Validation</div>
                <div class="step-desc">Immediate focus, illumination, and gradability verification.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_w3:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-icon">🧠</div>
                <div class="step-title">3. Explainable AI Grading</div>
                <div class="step-desc">DR classification with Grad-CAM heatmaps & lesion localization.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_w4:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-icon">🩺</div>
                <div class="step-title">4. Clinical Triage & Report</div>
                <div class="step-desc">PDF summary generated for tele-ophthalmologist validation.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(f"ℹ️ **Clinical Safety Disclaimer:** {CLINICAL_DISCLAIMER}")


def main() -> None:
    """Main application navigation controller."""
    initialize_session_state()

    # Sidebar Navigation
    st.sidebar.markdown(
        f"""
        <div style="padding: 10px 0 16px 0; border-bottom: 1px solid #334155;">
            <div style="font-size: 20px; font-weight: 800; color: #38bdf8;">👁️ {APP_NAME}</div>
            <div style="font-size: 11px; color: #94a3b8; margin-top: 2px;">Tele-Ophthalmology Screening</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    nav_options = [
        "1. Overview & Landing",
        "2. Patient Screening",
        "3. Quality & Structures",
        "4. Explainable AI (Grad-CAM)",
        "5. Clinical Report & PDF",
        "6. District Dashboard",
        "7. Rural Tele-Simulation",
    ]
    
    # Check if a navigation state was requested via button
    default_index = 0
    if "nav_selection" in st.session_state:
        selected_nav = st.session_state["nav_selection"]
        if selected_nav in nav_options:
            default_index = nav_options.index(selected_nav)
            
    selection = st.sidebar.radio(
        "Navigation Menu",
        options=nav_options,
        index=default_index,
        label_visibility="collapsed"
    )
    st.session_state["nav_selection"] = selection
    
    # Sidebar status footer
    st.sidebar.markdown("---")
    if is_using_real_pipeline():
        st.sidebar.success("🟢 AI Pipeline: Connected")
    else:
        st.sidebar.warning("🟡 AI Pipeline: Mock Fallback")
        
    st.sidebar.caption("Rural Telemedicine Screening Module\nVersion 0.1.0 (Phase 7)")

    # Route to selected page
    if selection == "1. Overview & Landing":
        render_home_page()
    elif selection == "2. Patient Screening":
        render_screening_page()
    elif selection == "3. Quality & Structures":
        render_analysis_page()
    elif selection == "4. Explainable AI (Grad-CAM)":
        render_explainability_page()
    elif selection == "5. Clinical Report & PDF":
        render_report_page()
    elif selection == "6. District Dashboard":
        render_dashboard_page()
    elif selection == "7. Rural Tele-Simulation":
        render_simulation_page()


if __name__ == "__main__":
    main()
