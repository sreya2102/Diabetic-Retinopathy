"""
Unit Tests for Phase 7: SimPy Discrete-Event Rural Telemedicine Simulation
"""

import pytest
from src.simulation.capacity import ScreeningProgramParams, CapacityEstimator, CapacityReport
from src.simulation.screening_queue import DiscreteEventScreeningSimulation


def test_simpy_simulation_execution():
    """Verify that discrete-event screening simulation runs through an 8-hour clinical day."""
    sim = DiscreteEventScreeningSimulation(
        num_phcs=5,
        patients_per_phc_day=10,
        avg_bandwidth_mbps=4.0,
        avg_image_size_mb=3.0,
        ai_inference_time_sec=1.0,
        num_ophthalmologists=2,
        doctor_review_time_min=2.0,
        sim_duration_hours=8.0
    )
    tasks = sim.run_simulation()
    
    assert len(tasks) > 0
    # Every task should have recorded completion
    assert all(t.total_turnaround_time_min > 0.0 for t in tasks)


def test_capacity_estimator_simulation():
    """Verify full capacity estimation and metrics output."""
    params = ScreeningProgramParams(
        num_phcs=20,
        patients_per_phc_per_day=20,
        working_days_per_year=250,
        avg_image_size_mb=3.5,
        avg_bandwidth_mbps=2.0,
        ai_inference_time_seconds=1.5,
        referral_rate=0.18,
        num_ophthalmologists=4,
        doctor_review_time_mins=3.0
    )
    report: CapacityReport = CapacityEstimator.run_discrete_event_simulation(params)
    
    assert report.daily_patients_processed > 0
    assert report.annual_projected_capacity > 0
    assert 0.0 <= report.ai_utilization_pct <= 100.0
    assert 0.0 <= report.doctor_utilization_pct <= 100.0
    assert report.avg_total_tat_min > 0.0
    assert report.daily_network_volume_gb > 0.0


def test_doctor_bottleneck_detection():
    """Verify that understaffing tele-ophthalmologists correctly triggers a doctor review bottleneck alert."""
    params = ScreeningProgramParams(
        num_phcs=30,
        patients_per_phc_per_day=30,  # 900 daily patients -> ~162 referable cases
        working_days_per_year=250,
        num_ophthalmologists=1,        # Only 1 doctor for 162 referable cases (8 hours = 480 mins / 162 = ~3 mins max)
        doctor_review_time_mins=5.0    # 162 * 5 = 810 mins needed (> 100% saturation)
    )
    report = CapacityEstimator.run_discrete_event_simulation(params)
    assert report.doctor_utilization_pct >= 85.0
    assert "Ophthalmologist" in report.primary_bottleneck or any("Ophthalmologist" in b for b in report.bottlenecks_identified)


def test_bandwidth_bottleneck_detection():
    """Verify that severe bandwidth throttling (< 1.0 Mbps) flags network constraints."""
    params = ScreeningProgramParams(
        num_phcs=10,
        patients_per_phc_per_day=15,
        avg_bandwidth_mbps=0.3,       # Very slow 300 Kbps uplink
        avg_image_size_mb=6.0,        # Large 6 MB images
        num_ophthalmologists=5
    )
    report = CapacityEstimator.run_discrete_event_simulation(params)
    assert report.avg_upload_time_sec > 45.0
    assert "Bandwidth" in report.primary_bottleneck or any("bandwidth" in b.lower() for b in report.bottlenecks_identified)
