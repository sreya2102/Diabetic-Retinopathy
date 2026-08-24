"""
Page: District-Level Tele-Screening Dashboard (Phase 6)
Aggregates operational metrics, patient throughput, referral triage, and clinician workload for rural health networks.
"""

from typing import Dict, Any, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.utils.session import initialize_session_state
from app.components.status_cards import render_mock_badge
from app.components.metrics import render_metrics_row


def get_district_demo_data(timeframe: str = "Last 14 Days", phc_filter: str = "All PHCs") -> Dict[str, Any]:
    """Generate structured operational analytics data for the district dashboard."""
    # Base volume multiplier based on timeframe
    mult = {"Last 7 Days": 0.5, "Last 14 Days": 1.0, "Last 30 Days": 2.1, "Year-to-Date (YTD)": 8.5}.get(timeframe, 1.0)
    
    total_screened = int(12480 * mult / 8.5) if timeframe != "Year-to-Date (YTD)" else 12480
    gradable_rate = 93.4
    referral_rate = 18.8
    urgent_rate = 2.5
    
    referable_count = int(total_screened * (referral_rate / 100.0))
    urgent_count = int(total_screened * (urgent_rate / 100.0))
    recaptures = int(total_screened * ((100.0 - gradable_rate) / 100.0))

    # PHC network breakdown
    phc_list = [
        {"phc": "Meenangadi CHC", "patients": int(3420 * mult / 8.5) if timeframe != "Year-to-Date (YTD)" else 3420, "gradable": "94.2%", "referrals": int(645 * mult / 8.5) if timeframe != "Year-to-Date (YTD)" else 645, "rate": 18.9, "bandwidth": "2.0 Mbps", "status": "Normal"},
        {"phc": "Sulthan Bathery GH", "patients": int(2980 * mult / 8.5) if timeframe != "Year-to-Date (YTD)" else 2980, "gradable": "92.8%", "referrals": int(560 * mult / 8.5) if timeframe != "Year-to-Date (YTD)" else 560, "rate": 18.8, "bandwidth": "4.0 Mbps", "status": "Normal"},
        {"phc": "Mananthavady DH", "patients": int(2450 * mult / 8.5) if timeframe != "Year-to-Date (YTD)" else 2450, "gradable": "95.1%", "referrals": int(472 * mult / 8.5) if timeframe != "Year-to-Date (YTD)" else 472, "rate": 19.3, "bandwidth": "8.0 Mbps", "status": "Clear"},
        {"phc": "Kalpetta FHC", "patients": int(2110 * mult / 8.5) if timeframe != "Year-to-Date (YTD)" else 2110, "gradable": "91.9%", "referrals": int(398 * mult / 8.5) if timeframe != "Year-to-Date (YTD)" else 398, "rate": 18.9, "bandwidth": "1.5 Mbps", "status": "Normal"},
        {"phc": "Vythiri CHC", "patients": int(1520 * mult / 8.5) if timeframe != "Year-to-Date (YTD)" else 1520, "gradable": "93.0%", "referrals": int(266 * mult / 8.5) if timeframe != "Year-to-Date (YTD)" else 266, "rate": 17.5, "bandwidth": "0.8 Mbps", "status": "Queue Backlog (Bandwidth)"},
    ]

    if phc_filter != "All PHCs":
        phc_list = [p for p in phc_list if p["phc"] == phc_filter]
        if phc_list:
            total_screened = phc_list[0]["patients"]
            referable_count = phc_list[0]["referrals"]
            urgent_count = int(total_screened * 0.025)
            recaptures = int(total_screened * 0.07)

    return {
        "total_screened": total_screened,
        "gradable_rate": gradable_rate,
        "referable_count": referable_count,
        "urgent_count": urgent_count,
        "recaptures": recaptures,
        "phc_data": phc_list
    }


def render_dashboard_page() -> None:
    """Render the district-level operational analytics dashboard."""
    initialize_session_state()
    
    st.markdown("## 📊 District-Level Tele-Screening Dashboard")
    st.caption("Real-time operational monitoring across primary health centres (PHCs) and tele-ophthalmology reading hubs.")
    
    render_mock_badge()

    # Interactive Filter Controls
    col_f1, col_f2, col_f3 = st.columns([2, 2, 2])
    with col_f1:
        timeframe = st.selectbox(
            "Timeframe Horizon",
            options=["Last 7 Days", "Last 14 Days", "Last 30 Days", "Year-to-Date (YTD)"],
            index=1
        )
    with col_f2:
        phc_filter = st.selectbox(
            "Primary Health Centre (PHC)",
            options=["All PHCs", "Meenangadi CHC", "Sulthan Bathery GH", "Mananthavady DH", "Kalpetta FHC", "Vythiri CHC"],
            index=0
        )
    with col_f3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption(f"Showing live screening telemetry for **{phc_filter}** ({timeframe}).")

    data = get_district_demo_data(timeframe=timeframe, phc_filter=phc_filter)

    st.markdown("---")

    # 1. Top-Level Operational Metrics
    st.markdown("### 1. Key Screening Performance Indicators (KPIs)")
    render_metrics_row([
        {
            "label": "Total Patients Screened",
            "value": f"{data['total_screened']:,}",
            "delta": "+14.2% MoM",
            "subtext": "Active across rural network",
        },
        {
            "label": "Image Gradability Rate",
            "value": f"{data['gradable_rate']:.1f}%",
            "delta": "+2.1%",
            "subtext": f"{data['recaptures']:,} recaptures requested",
            "border_color": "#10b981"
        },
        {
            "label": "Referable DR Cases",
            "value": f"{data['referable_count']:,}",
            "delta": "18.8% referral rate",
            "delta_color": "#f59e0b",
            "subtext": "Grade ≥ 2 triage threshold",
        },
        {
            "label": "Urgent Triage (PDR)",
            "value": f"{data['urgent_count']:,}",
            "delta": "2.5%",
            "delta_color": "#dc2626",
            "subtext": "< 2-week specialist consult",
            "border_color": "#fca5a5"
        },
    ])
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(
            """
            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px;">
                <span style="font-size: 11px; color: #64748b; font-weight: 700; text-transform: uppercase;">AVERAGE AI INFERENCE TIME</span>
                <div style="font-size: 20px; font-weight: 700; color: #0284c7;">1.42 seconds <span style="font-size: 12px; color: #64748b; font-weight: 400;">/ fundus image</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_t2:
        st.markdown(
            """
            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px;">
                <span style="font-size: 11px; color: #64748b; font-weight: 700; text-transform: uppercase;">AVERAGE OPHTHALMOLOGIST REVIEW</span>
                <div style="font-size: 20px; font-weight: 700; color: #059669;">2.8 minutes <span style="font-size: 12px; color: #64748b; font-weight: 400;">/ flagged referable case</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # 2. Charts: Severity Distribution & Screening Funnel
    st.markdown("### 2. Clinical Distribution & Tele-Screening Pipeline Funnel")
    col_ch1, col_ch2 = st.columns(2)
    
    with col_ch1:
        # Severity Donut Chart
        severity_df = pd.DataFrame({
            "Severity Grade": [
                "Grade 0: No DR",
                "Grade 1: Mild NPDR",
                "Grade 2: Moderate NPDR",
                "Grade 3: Severe NPDR",
                "Grade 4: Proliferative DR"
            ],
            "Patient Count": [
                int(data["total_screened"] * 0.66),
                int(data["total_screened"] * 0.15),
                int(data["total_screened"] * 0.11),
                int(data["total_screened"] * 0.055),
                int(data["total_screened"] * 0.025)
            ]
        })
        
        fig_donut = px.pie(
            severity_df,
            names="Severity Grade",
            values="Patient Count",
            hole=0.45,
            color="Severity Grade",
            color_discrete_map={
                "Grade 0: No DR": "#10b981",
                "Grade 1: Mild NPDR": "#3b82f6",
                "Grade 2: Moderate NPDR": "#f59e0b",
                "Grade 3: Severe NPDR": "#ea580c",
                "Grade 4: Proliferative DR": "#dc2626"
            },
            title=f"DR Severity Breakdown (N={data['total_screened']:,})"
        )
        fig_donut.update_layout(
            margin=dict(l=10, r=10, t=35, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_ch2:
        # Screening Pipeline Funnel
        funnel_df = pd.DataFrame({
            "Stage": [
                "1. Registered at PHC",
                "2. Image Acquired",
                "3. Gradability Passed",
                "4. AI Inferred",
                "5. Referable Triage (Grade ≥ 2)",
                "6. Specialist Validated"
            ],
            "Count": [
                data["total_screened"],
                data["total_screened"],
                int(data["total_screened"] * 0.934),
                int(data["total_screened"] * 0.934),
                data["referable_count"],
                int(data["referable_count"] * 0.92)
            ]
        })
        fig_funnel = px.funnel(
            funnel_df,
            x="Count",
            y="Stage",
            title="Screening Triage Pipeline Progression",
            color_discrete_sequence=["#0284c7"]
        )
        fig_funnel.update_layout(margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig_funnel, use_container_width=True)

    st.markdown("---")

    # 3. Charts: Throughput Velocity & Specialist Workload
    st.markdown("### 3. Screening Velocity & Tele-Ophthalmologist Workload")
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        # Daily throughput trend
        num_days = 14 if timeframe == "Last 14 Days" else (7 if timeframe == "Last 7 Days" else 30)
        dates = pd.date_range(end=pd.Timestamp.today(), periods=num_days, freq='D')
        daily_base = max(int(data["total_screened"] / num_days), 10)
        
        throughput_df = pd.DataFrame({
            "Date": dates.strftime('%b %d'),
            "Routine Screenings (Grade 0-1)": [int(daily_base * 0.81 + (i % 4) * 5) for i in range(num_days)],
            "Referrals Flagged (Grade ≥ 2)": [int(daily_base * 0.19 + (i % 3) * 2) for i in range(num_days)]
        })
        
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=throughput_df["Date"],
            y=throughput_df["Routine Screenings (Grade 0-1)"],
            name="Non-Referable",
            marker_color="#38bdf8"
        ))
        fig_bar.add_trace(go.Bar(
            x=throughput_df["Date"],
            y=throughput_df["Referrals Flagged (Grade ≥ 2)"],
            name="Referral Required",
            marker_color="#f87171"
        ))
        fig_bar.update_layout(
            barmode='stack',
            title=f"Screening Velocity ({timeframe})",
            margin=dict(l=10, r=10, t=35, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_v2:
        # Specialist Workload
        doc_df = pd.DataFrame({
            "Ophthalmologist": ["Dr. K. Menon (Retina Hub)", "Dr. S. Nair (District Hospital)", "Dr. A. Sharma (Tele-Reader)"],
            "Cases Reviewed": [int(data["referable_count"] * 0.42), int(data["referable_count"] * 0.35), int(data["referable_count"] * 0.23)],
            "Pending Queue": [14, 18, 9]
        })
        fig_doc = px.bar(
            doc_df,
            x="Ophthalmologist",
            y=["Cases Reviewed", "Pending Queue"],
            barmode="group",
            title="Tele-Ophthalmologist Review Allocation",
            color_discrete_map={"Cases Reviewed": "#10b981", "Pending Queue": "#f59e0b"}
        )
        fig_doc.update_layout(
            margin=dict(l=10, r=10, t=35, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_doc, use_container_width=True)

    st.markdown("---")

    # 4. PHC Center Breakdown Table
    st.markdown("### 4. Primary Health Centre (PHC) Network Operational Status")
    table_df = pd.DataFrame(data["phc_data"])
    table_df = table_df.rename(columns={
        "phc": "Primary Health Centre",
        "patients": "Patients Screened",
        "gradable": "Gradability %",
        "referrals": "Referrals (Grade ≥ 2)",
        "rate": "Referral Rate %",
        "bandwidth": "Avg Bandwidth",
        "status": "Uplink & Queue Status"
    })
    st.dataframe(table_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    render_dashboard_page()
