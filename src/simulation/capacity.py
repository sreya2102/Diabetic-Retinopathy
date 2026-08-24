"""
Rural Capacity & Bottleneck Analytics Engine (Phase 7)
Computes operational throughput, clinician workload capacity, and discrete-event SimPy analytics.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from src.simulation.screening_queue import DiscreteEventScreeningSimulation, ScreeningTaskRecord


@dataclass
class ScreeningProgramParams:
    """Configuration parameters for district-scale rural screening simulation."""
    num_phcs: int = 20
    patients_per_phc_per_day: int = 20
    working_days_per_year: int = 250
    avg_image_size_mb: float = 3.5
    avg_bandwidth_mbps: float = 2.0
    ai_inference_time_seconds: float = 2.0
    referral_rate: float = 0.18  # 18% referable DR prevalence
    num_ophthalmologists: int = 3
    doctor_review_time_mins: float = 3.0  # 3 minutes per referable case


@dataclass
class CapacityReport:
    """Consolidated annual capacity and bottleneck diagnostic report."""
    annual_target_patients: int
    annual_projected_capacity: int
    daily_patients_processed: int
    referable_patients_daily: int
    
    # Resource Utilization %
    ai_utilization_pct: float
    network_utilization_pct: float
    doctor_utilization_pct: float
    
    # Latencies in minutes
    avg_upload_time_sec: float
    avg_ai_wait_min: float
    avg_doctor_wait_min: float
    avg_total_tat_min: float
    
    # Daily network transfer volume
    daily_network_volume_gb: float
    
    # Bottleneck identification
    primary_bottleneck: str
    bottlenecks_identified: List[str]

    @property
    def daily_district_volume(self) -> int:
        """Alias for daily_patients_processed."""
        return self.daily_patients_processed

    @property
    def annual_screened_patients(self) -> int:
        """Alias for annual_projected_capacity."""
        return self.annual_projected_capacity


class CapacityEstimator:
    """Simulates district-scale screening programs and performs bottleneck root cause analysis."""
    
    @staticmethod
    def estimate_annual_capacity(params: ScreeningProgramParams) -> CapacityReport:
        """Estimate annual screening capacity and identify bottlenecks."""
        return CapacityEstimator.run_discrete_event_simulation(params)

    @staticmethod
    def run_discrete_event_simulation(params: ScreeningProgramParams) -> CapacityReport:
        """Run SimPy discrete-event simulation and synthesize capacity & bottleneck metrics."""
        sim = DiscreteEventScreeningSimulation(
            num_phcs=params.num_phcs,
            patients_per_phc_day=params.patients_per_phc_per_day,
            avg_bandwidth_mbps=params.avg_bandwidth_mbps,
            avg_image_size_mb=params.avg_image_size_mb,
            ai_inference_time_sec=params.ai_inference_time_seconds,
            referral_rate=params.referral_rate,
            num_ophthalmologists=params.num_ophthalmologists,
            doctor_review_time_min=params.doctor_review_time_mins,
            sim_duration_hours=8.0
        )
        
        tasks: List[ScreeningTaskRecord] = sim.run_simulation()
        
        daily_processed = len(tasks)
        referrals = [t for t in tasks if t.is_referable]
        daily_referrals = len(referrals)
        
        # Scaling to annual capacity
        annual_capacity = daily_processed * params.working_days_per_year
        
        # Average latencies
        avg_upload_sec = float(np.mean([t.upload_duration_min * 60.0 for t in tasks])) if tasks else 0.0
        avg_ai_wait = float(np.mean([t.ai_wait_min for t in tasks])) if tasks else 0.0
        avg_doc_wait = float(np.mean([t.doctor_wait_min for t in referrals])) if referrals else 0.0
        avg_tat = float(np.mean([t.total_turnaround_time_min for t in tasks])) if tasks else 0.0
        
        # Resource Utilization
        total_sim_mins = 8.0 * 60.0  # 480 minutes
        # AI capacity (2 workers * 480 mins = 960 worker minutes)
        ai_util = min((sim.total_ai_busy_minutes / (2.0 * total_sim_mins)) * 100.0, 100.0)
        # Doctor capacity (num_doctors * 480 mins)
        total_doc_capacity_mins = max(params.num_ophthalmologists, 1) * total_sim_mins
        doc_util = min((sim.total_doctor_busy_minutes / total_doc_capacity_mins) * 100.0, 100.0)
        # Network channel utilization
        net_util = min((sim.total_upload_busy_minutes / (params.num_phcs * total_sim_mins)) * 100.0, 100.0)
        
        daily_gb = (daily_processed * params.avg_image_size_mb) / 1024.0

        # Bottleneck detection
        bottlenecks = []
        primary_bottleneck = "Optimal Flow — No Major Bottleneck"
        
        if doc_util >= 85.0 or avg_doc_wait > 30.0:
            bottlenecks.append(f"Ophthalmologist reading capacity saturated ({doc_util:.1f}% util, {avg_doc_wait:.1f}m avg queue wait).")
            primary_bottleneck = "Ophthalmologist Tele-Review Capacity"
        elif params.avg_bandwidth_mbps < 1.0 or avg_upload_sec > 45.0:
            bottlenecks.append(f"PHC uplink bandwidth constrained ({params.avg_bandwidth_mbps:.1f} Mbps; {avg_upload_sec:.1f}s/image).")
            primary_bottleneck = "Rural Network Bandwidth Uplink"
        elif ai_util >= 90.0 or avg_ai_wait > 10.0:
            bottlenecks.append(f"AI GPU inference server queue backlogged ({ai_util:.1f}% util).")
            primary_bottleneck = "AI Inference Server Capacity"
        elif daily_processed < (params.num_phcs * params.patients_per_phc_per_day * 0.85):
            bottlenecks.append("PHC patient acquisition throughput throttled by local intake delays.")
            primary_bottleneck = "Local Camera Acquisition Speed"

        return CapacityReport(
            annual_target_patients=100000,
            annual_projected_capacity=annual_capacity,
            daily_patients_processed=daily_processed,
            referable_patients_daily=daily_referrals,
            ai_utilization_pct=round(ai_util, 1),
            network_utilization_pct=round(net_util, 1),
            doctor_utilization_pct=round(doc_util, 1),
            avg_upload_time_sec=round(avg_upload_sec, 1),
            avg_ai_wait_min=round(avg_ai_wait, 2),
            avg_doctor_wait_min=round(avg_doc_wait, 1),
            avg_total_tat_min=round(avg_tat, 1),
            daily_network_volume_gb=round(daily_gb, 2),
            primary_bottleneck=primary_bottleneck,
            bottlenecks_identified=bottlenecks
        )
