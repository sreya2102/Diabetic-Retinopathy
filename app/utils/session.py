"""
Session State Management for RETINASCAN
Manages patient information, image buffers, screening states, and analysis cache safely.
"""

from typing import Any, Dict, Optional
import streamlit as st


def initialize_session_state() -> None:
    """Initialize default session state keys if they are not already present."""
    defaults: Dict[str, Any] = {
        # Patient demographics
        "patient_id": "",
        "patient_age": None,
        "patient_gender": "Unspecified",
        "district": "",
        "phc_center": "",
        
        # Image acquisition data
        "uploaded_image_bytes": None,
        "uploaded_image_filename": None,
        "uploaded_image_format": None,
        "uploaded_image_size_kb": 0.0,
        
        # Screening analysis results
        "analysis_result": None,
        "is_mock_result": True,
        
        # UI navigation & workflow state
        "current_step": 1,
        "screening_timestamp": None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_patient_data(
    patient_id: str,
    age: Optional[int] = None,
    gender: str = "Unspecified",
    district: str = "",
    phc_center: str = "",
) -> None:
    """Store patient demographic data in session state."""
    st.session_state["patient_id"] = patient_id.strip()
    st.session_state["patient_age"] = age
    st.session_state["patient_gender"] = gender
    st.session_state["district"] = district.strip()
    st.session_state["phc_center"] = phc_center.strip()


def get_patient_data() -> Dict[str, Any]:
    """Retrieve patient demographic data from session state."""
    return {
        "patient_id": st.session_state.get("patient_id", ""),
        "patient_age": st.session_state.get("patient_age"),
        "patient_gender": st.session_state.get("patient_gender", "Unspecified"),
        "district": st.session_state.get("district", ""),
        "phc_center": st.session_state.get("phc_center", ""),
    }


def set_uploaded_image(
    image_bytes: bytes,
    filename: str,
    image_format: str,
    size_kb: float = 0.0,
) -> None:
    """Store image payload and metadata into session state."""
    st.session_state["uploaded_image_bytes"] = image_bytes
    st.session_state["uploaded_image_filename"] = filename
    st.session_state["uploaded_image_format"] = image_format
    st.session_state["uploaded_image_size_kb"] = size_kb


def set_analysis_result(result: Dict[str, Any]) -> None:
    """Store the structured analysis result dictionary into session state."""
    st.session_state["analysis_result"] = result
    st.session_state["is_mock_result"] = bool(result.get("is_mock", True))


def get_analysis_result() -> Optional[Dict[str, Any]]:
    """Retrieve the cached analysis result if available."""
    return st.session_state.get("analysis_result", None)


def clear_screening() -> None:
    """Reset the current screening session to blank defaults."""
    st.session_state["patient_id"] = ""
    st.session_state["patient_age"] = None
    st.session_state["patient_gender"] = "Unspecified"
    st.session_state["district"] = ""
    st.session_state["phc_center"] = ""
    st.session_state["uploaded_image_bytes"] = None
    st.session_state["uploaded_image_filename"] = None
    st.session_state["uploaded_image_format"] = None
    st.session_state["uploaded_image_size_kb"] = 0.0
    st.session_state["analysis_result"] = None
    st.session_state["is_mock_result"] = True
    st.session_state["current_step"] = 1
    st.session_state["screening_timestamp"] = None
