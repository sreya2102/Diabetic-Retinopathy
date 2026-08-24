"""
Fundus Image Viewer Component
Displays uploaded retinal images, original vs enhanced comparisons, and acquisition details.
"""

from typing import Any, Optional
from PIL import Image
import streamlit as st
import io


def render_fundus_viewer(
    image_data: Any,
    caption: str = "Fundus Image",
    use_column_width: bool = True,
    metadata: Optional[dict] = None
) -> None:
    """
    Render a retinal fundus image with medical framing and optional metadata.
    Accepts PIL Image, bytes, or numpy array.
    """
    if image_data is None:
        st.info("No fundus image loaded.")
        return
        
    try:
        if isinstance(image_data, bytes):
            img = Image.open(io.BytesIO(image_data))
        elif isinstance(image_data, Image.Image):
            img = image_data
        else:
            img = image_data  # numpy array or streamlit-compatible
            
        st.image(img, caption=caption, use_container_width=use_column_width)
        
        if metadata:
            meta_items = [f"<b>{k}:</b> {v}" for k, v in metadata.items()]
            st.markdown(
                f"<div style='font-size: 12px; color: #64748b; margin-top: 4px;'>"
                f"{' &bull; '.join(meta_items)}"
                f"</div>",
                unsafe_allow_html=True
            )
    except Exception as e:
        st.error(f"Error rendering fundus image: {str(e)}")


def render_side_by_side_comparison(
    left_image: Any,
    right_image: Any,
    left_title: str = "Original Fundus",
    right_title: str = "Enhanced Image",
    right_placeholder_text: str = "Enhancement pipeline output will appear here."
) -> None:
    """Render side-by-side comparison between original and processed fundus images."""
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{left_title}**")
        render_fundus_viewer(left_image, caption=left_title)
        
    with col2:
        st.markdown(f"**{right_title}**")
        if right_image is not None:
            render_fundus_viewer(right_image, caption=right_title)
        else:
            st.markdown(
                f"""
                <div style="
                    border: 2px dashed #cbd5e1;
                    border-radius: 8px;
                    padding: 48px 16px;
                    text-align: center;
                    color: #64748b;
                    background: #f8fafc;
                ">
                    <div style="font-size: 24px; margin-bottom: 8px;">🔬</div>
                    <div style="font-size: 14px; font-weight: 600;">{right_title}</div>
                    <div style="font-size: 12px; color: #94a3b8; margin-top: 4px;">{right_placeholder_text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
