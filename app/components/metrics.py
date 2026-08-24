"""
Metric Card Components for Clinical and Operational Dashboards
Renders clean medical KPI cards with labels, values, and optional deltas/subtexts.
"""

from typing import Optional
import streamlit as st


def render_metric_card(
    label: str,
    value: str,
    subtext: Optional[str] = None,
    delta: Optional[str] = None,
    delta_color: str = "#10b981",
    border_color: str = "#e2e8f0",
) -> None:
    """Render a single clinical metric tile."""
    subtext_html = f"<div style='font-size: 12px; color: #64748b; margin-top: 4px;'>{subtext}</div>" if subtext else ""
    delta_html = f"<span style='font-size: 12px; color: {delta_color}; font-weight: 600; margin-left: 6px;'>{delta}</span>" if delta else ""
    
    st.markdown(
        f"""
        <div style="
            background: #ffffff;
            border: 1px solid {border_color};
            border-radius: 8px;
            padding: 16px 18px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            margin-bottom: 12px;
        ">
            <div style="font-size: 12px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px;">
                {label}
            </div>
            <div style="font-size: 24px; font-weight: 700; color: #0f172a; margin-top: 4px; display: flex; align-items: baseline;">
                {value} {delta_html}
            </div>
            {subtext_html}
        </div>
        """,
        unsafe_allow_html=True
    )


def render_metrics_row(metrics: list[dict]) -> None:
    """Render a horizontal row of metric cards using Streamlit columns."""
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            render_metric_card(
                label=m.get("label", ""),
                value=m.get("value", ""),
                subtext=m.get("subtext"),
                delta=m.get("delta"),
                delta_color=m.get("delta_color", "#10b981"),
                border_color=m.get("border_color", "#e2e8f0"),
            )
