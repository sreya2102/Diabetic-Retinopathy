"""
Page: Rural Telemedicine & Bandwidth Simulation (Phase 7)
Simulates end-to-end multi-PHC screening, network constraints, AI throughput, and ophthalmologist review capacity.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.utils.session import initialize_session_state
from app.components.status_cards import render_mock_badge
from app.components.metrics import render_metrics_row
from src.simulation.capacity import ScreeningProgramParams, CapacityEstimator, CapacityReport


def render_simulation_page() -> None:
    """Render the interactive SimPy rural telemedicine simulation page."""
    initialize_session_state()
    
    st.markdown("## 📡 Rural Telemedicine Network & Capacity Simulation")
    st.caption("Discrete-event SimPy simulation modeling PHC acquisition, bandwidth throttling, AI server inference, and tele-ophthalmology triage.")
    
    render_mock_badge()

    # Simulation Parameter Controls
    st.markdown("### 1. Telemedicine Network & Resource Parameters")
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("##### 🏥 Rural Clinic Network")
            num_phcs = st.slider("Number of Rural PHCs", min_value=5, max_value=50, value=20, step=1)
            patients_per_phc = st.slider("Patients per PHC per Day", min_value=5, max_value=60, value=20, step=1)
            
        with col2:
            st.markdown("##### 🌐 Network & Image Transfer")
            avg_bandwidth = st.slider("PHC Uplink Bandwidth (Mbps)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
            avg_image_size = st.slider("Average Fundus Image Size (MB)", min_value=1.0, max_value=10.0, value=3.5, step=0.5)
            
        with col3:
            st.markdown("##### 🧠 AI & Tele-Ophthalmologist Review")
            ai_time_sec = st.slider("AI Inference Time (seconds)", min_value=0.5, max_value=8.0, value=1.5, step=0.5)
            num_doctors = st.slider("Tele-Ophthalmologists on Duty", min_value=1, max_value=10, value=3, step=1)
            doc_review_mins = st.slider("Doctor Review Time per Case (mins)", min_value=1.0, max_value=8.0, value=3.0, step=0.5)

    params = ScreeningProgramParams(
        num_phcs=num_phcs,
        patients_per_phc_per_day=patients_per_phc,
        working_days_per_year=250,
        avg_image_size_mb=avg_image_size,
        avg_bandwidth_mbps=avg_bandwidth,
        ai_inference_time_seconds=ai_time_sec,
        referral_rate=0.18,
        num_ophthalmologists=num_doctors,
        doctor_review_time_mins=doc_review_mins
    )

    st.markdown("<br>", unsafe_allow_html=True)
    run_sim = st.button("⚡ Run Rural Telemedicine Simulation (SimPy)", type="primary", use_container_width=True)

    # Run discrete-event simulation
    with st.spinner("Running SimPy discrete-event simulation across multi-PHC network channels..."):
        report: CapacityReport = CapacityEstimator.run_discrete_event_simulation(params)

    st.markdown("---")

    # 2. Key Simulation Output Metrics
    st.markdown("### 2. Simulation Results & Capacity Telemetry")
    
    cap_color = "#10b981" if report.annual_projected_capacity >= 100000 else "#ea580c"
    render_metrics_row([
        {
            "label": "Daily Patients Screened",
            "value": f"{report.daily_patients_processed:,}",
            "subtext": f"{report.referable_patients_daily:,} referable cases flagged",
        },
        {
            "label": "Projected Annual Capacity",
            "value": f"{report.annual_projected_capacity:,}",
            "subtext": "Target: 100,000+ patients/year",
            "border_color": cap_color
        },
        {
            "label": "Avg End-to-End TAT",
            "value": f"{report.avg_total_tat_min:.1f} mins",
            "subtext": "PHC arrival to diagnostic report",
            "border_color": "#0284c7"
        },
        {
            "label": "Daily Network Load",
            "value": f"{report.daily_network_volume_gb:.2f} GB",
            "subtext": f"Avg upload: {report.avg_upload_time_sec:.1f}s / image",
        }
    ])

    # Bottleneck Diagnostic Alert Card
    is_bottleneck = report.primary_bottleneck != "Optimal Flow — No Major Bottleneck"
    alert_bg = "#fef2f2" if is_bottleneck else "#ecfdf5"
    alert_border = "#ef4444" if is_bottleneck else "#10b981"
    alert_color = "#991b1b" if is_bottleneck else "#065f46"
    
    st.markdown(
        f"""
        <div style="
            background-color: {alert_bg};
            border: 1px solid {alert_border}40;
            border-left: 6px solid {alert_border};
            border-radius: 8px;
            padding: 16px 20px;
            margin: 16px 0;
        ">
            <div style="font-size: 11px; font-weight: 700; color: {alert_color}; text-transform: uppercase; letter-spacing: 0.5px;">
                Diagnostic Bottleneck Analysis
            </div>
            <div style="font-size: 20px; font-weight: 700; color: #1e293b; margin: 4px 0;">
                Primary Constraint: {report.primary_bottleneck}
            </div>
            <div style="font-size: 13px; color: #475569;">
                {' &bull; '.join(report.bottlenecks_identified) if report.bottlenecks_identified else 'All screening pipeline stages operating smoothly within capacity limits.'}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # 3. Visualizations: Resource Utilization & Pipeline Delays
    st.markdown("### 3. Resource Saturation & Queue Delays")
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        # Resource Utilization Bar Chart
        util_df = pd.DataFrame({
            "Resource": ["AI Server GPU", "PHC Network Uplink", "Ophthalmologist Reading Team"],
            "Utilization %": [report.ai_utilization_pct, report.network_utilization_pct, report.doctor_utilization_pct],
            "Threshold": [85.0, 85.0, 85.0]
        })
        
        fig_util = px.bar(
            util_df,
            x="Resource",
            y="Utilization %",
            color="Utilization %",
            color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
            range_color=[0, 100],
            title="Screening Pipeline Resource Utilization (%)"
        )
        fig_util.add_hline(y=85, line_dash="dash", line_color="#ef4444", annotation_text="85% Saturation Threshold")
        fig_util.update_layout(margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig_util, use_container_width=True)

    with col_g2:
        # Latency breakdown
        delay_df = pd.DataFrame({
            "Stage": ["1. Camera Capture", "2. Network Transmission", "3. AI Inference Queue", "4. Doctor Review Queue"],
            "Time (Minutes)": [3.5, report.avg_upload_time_sec / 60.0, report.avg_ai_wait_min, report.avg_doctor_wait_min]
        })
        fig_delay = px.bar(
            delay_df,
            x="Stage",
            y="Time (Minutes)",
            title="Average Pipeline Queue Latency Breakdown",
            color_discrete_sequence=["#0284c7"]
        )
        fig_delay.update_layout(margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig_delay, use_container_width=True)

    st.markdown("---")
    
    # 4. Annual 100k Target Progress Gauge
    st.markdown("### 4. Annual Scale: 100,000+ Target Feasibility")
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = report.annual_projected_capacity,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Annual Screening Throughput (Patients / Year)", 'font': {'size': 18}},
        delta = {'reference': 100000, 'increasing': {'color': "#10b981"}, 'decreasing': {'color': "#ef4444"}},
        gauge = {
            'axis': {'range': [None, 150000], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#0284c7"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#cbd5e1",
            'steps': [
                {'range': [0, 60000], 'color': '#fee2e2'},
                {'range': [60000, 100000], 'color': '#fef3c7'},
                {'range': [100000, 150000], 'color': '#ecfdf5'}
            ],
            'threshold': {
                'line': {'color': "green", 'width': 4},
                'thickness': 0.75,
                'value': 100000
            }
        }
    ))
    fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)


if __name__ == "__main__":
    render_simulation_page()
