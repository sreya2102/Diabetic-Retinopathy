"""
Status Card Components for Clinical Presentation
Renders clean medical status cards for Gradability, Referability, and DR Severity.
"""

from typing import Optional
import streamlit as st
from app.utils.formatting import get_dr_grade_info, format_referral_status, format_gradability


def render_mock_badge() -> None:
    """Render high-visibility warning banner when mock/demo results are active."""
    st.markdown(
        """
        <div style="
            background-color: #fffbeb;
            border-left: 4px solid #f59e0b;
            padding: 10px 16px;
            border-radius: 6px;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        ">
            <div style="font-size: 13px; color: #92400e; font-weight: 600; letter-spacing: 0.5px;">
                ⚠️ DEMO / MOCK RESULT — Real AI pipeline not connected. Demonstration view only.
            </div>
            <span style="
                background: #fef3c7;
                color: #b45309;
                font-size: 11px;
                font-weight: 700;
                padding: 2px 8px;
                border-radius: 4px;
                border: 1px solid #fcd34d;
            ">TEST MODE</span>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_dr_severity_card(class_id: int, confidence: Optional[float] = None) -> None:
    """Render clinical DR severity card with color-coded risk level."""
    info = get_dr_grade_info(class_id)
    conf_str = f" • Confidence: {confidence*100:.1f}%" if confidence is not None else ""
    
    st.markdown(
        f"""
        <div style="
            background-color: {info['bg_color']};
            border: 1px solid {info['color']}40;
            border-left: 6px solid {info['color']};
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 16px;
        ">
            <div style="font-size: 12px; font-weight: 700; color: {info['color']}; text-transform: uppercase; letter-spacing: 0.5px;">
                Predicted DR Severity (Grade {class_id})
            </div>
            <div style="font-size: 24px; font-weight: 700; color: #1e293b; margin: 4px 0;">
                {info['label']}
            </div>
            <div style="font-size: 14px; color: #475569;">
                {info['description']}{conf_str}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_referral_card(referable: bool) -> None:
    """Render clear clinical referral recommendation card."""
    label, color, bg = format_referral_status(referable)
    action_text = (
        "Immediate referral to an ophthalmology center required for clinical examination and management."
        if referable else
        "No referral required at this stage. Routine annual diabetic eye screening recommended."
    )
    
    st.markdown(
        f"""
        <div style="
            background-color: {bg};
            border: 1px solid {color}40;
            border-left: 6px solid {color};
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 16px;
        ">
            <div style="font-size: 12px; font-weight: 700; color: {color}; text-transform: uppercase; letter-spacing: 0.5px;">
                Screening Referral Decision
            </div>
            <div style="font-size: 20px; font-weight: 700; color: #1e293b; margin: 4px 0;">
                {label}
            </div>
            <div style="font-size: 14px; color: #475569;">
                {action_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_quality_card(gradable: bool, score: float = 0.0, feedback: Optional[list] = None) -> None:
    """Render image quality assessment card."""
    label, color, bg = format_gradability(gradable, score)
    
    feedback_html = ""
    if feedback:
        items = "".join([f"<li style='margin-bottom: 2px;'>{item}</li>" for item in feedback])
        feedback_html = f"<ul style='margin: 6px 0 0 16px; padding: 0; font-size: 13px; color: #475569;'>{items}</ul>"
        
    st.markdown(
        f"""
        <div style="
            background-color: {bg};
            border: 1px solid {color}40;
            border-left: 6px solid {color};
            border-radius: 8px;
            padding: 14px 18px;
            margin-bottom: 16px;
        ">
            <div style="font-size: 12px; font-weight: 700; color: {color}; text-transform: uppercase; letter-spacing: 0.5px;">
                Image Quality & Gradability
            </div>
            <div style="font-size: 18px; font-weight: 700; color: #1e293b; margin: 2px 0;">
                {label}
            </div>
            {feedback_html}
        </div>
        """,
        unsafe_allow_html=True
    )
