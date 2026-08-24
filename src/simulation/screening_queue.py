"""
Screening Queue & Discrete-Event Workflow Simulation Engine (Phase 7)
Uses SimPy to model multi-stage patient arrival, acquisition, upload, AI inference, and ophthalmologist review.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import random
import simpy

from src.simulation.rural_network import RuralNetworkSimulator


@dataclass
class ScreeningTaskRecord:
    """Detailed log record of a single patient task moving through the simulation pipeline."""
    task_id: str
    patient_id: str
    phc_id: str
    arrival_time_min: float
    acquisition_wait_min: float = 0.0
    acquisition_duration_min: float = 0.0
    upload_wait_min: float = 0.0
    upload_duration_min: float = 0.0
    ai_wait_min: float = 0.0
    ai_duration_min: float = 0.0
    is_referable: bool = False
    doctor_wait_min: float = 0.0
    doctor_duration_min: float = 0.0
    total_turnaround_time_min: float = 0.0


class DiscreteEventScreeningSimulation:
    """
    Complete SimPy discrete-event simulation engine for rural tele-ophthalmology networks.
    """
    
    def __init__(
        self,
        num_phcs: int = 20,
        patients_per_phc_day: int = 20,
        avg_bandwidth_mbps: float = 2.0,
        avg_image_size_mb: float = 3.5,
        ai_inference_time_sec: float = 2.0,
        referral_rate: float = 0.18,
        num_ophthalmologists: int = 3,
        doctor_review_time_min: float = 3.0,
        sim_duration_hours: float = 8.0  # 1 clinical day = 8 hours
    ):
        self.num_phcs = num_phcs
        self.patients_per_phc_day = patients_per_phc_day
        self.avg_bandwidth_mbps = avg_bandwidth_mbps
        self.avg_image_size_mb = avg_image_size_mb
        self.ai_inference_time_sec = ai_inference_time_sec
        self.referral_rate = referral_rate
        self.num_ophthalmologists = num_ophthalmologists
        self.doctor_review_time_min = doctor_review_time_min
        self.sim_duration_minutes = sim_duration_hours * 60.0
        
        self.env = simpy.Environment()
        self.network_sim = RuralNetworkSimulator(env=self.env)
        
        # Shared Central Resources
        self.ai_server_resource = simpy.Resource(self.env, capacity=2)  # Dual AI GPU inference workers
        self.doctor_resource = simpy.Resource(self.env, capacity=max(self.num_ophthalmologists, 1))
        
        # Output task logs
        self.completed_tasks: List[ScreeningTaskRecord] = []
        
        # Cumulative busy time counters (for utilization calculation)
        self.total_ai_busy_minutes = 0.0
        self.total_doctor_busy_minutes = 0.0
        self.total_upload_busy_minutes = 0.0

    def patient_screening_lifecycle(self, task: ScreeningTaskRecord):
        """SimPy process modeling the life of a single screening record."""
        # 1. Image Acquisition at PHC
        acq_start = self.env.now
        acq_duration = random.uniform(2.5, 4.5)  # 2.5 - 4.5 minutes to position patient and capture
        yield self.env.timeout(acq_duration)
        task.acquisition_duration_min = acq_duration

        # 2. Network Transmission from PHC to Cloud / Hub
        upload_start = self.env.now
        tx_sec = self.network_sim.calculate_transmission_time_seconds(
            self.avg_image_size_mb,
            self.avg_bandwidth_mbps
        )
        tx_min = tx_sec / 60.0
        yield self.env.timeout(tx_min)
        task.upload_duration_min = tx_min
        self.total_upload_busy_minutes += tx_min

        # 3. AI Quality & Severity Inference Queue
        ai_queue_start = self.env.now
        with self.ai_server_resource.request() as req:
            yield req
            task.ai_wait_min = self.env.now - ai_queue_start
            
            ai_dur_min = (self.ai_inference_time_sec * random.uniform(0.9, 1.1)) / 60.0
            yield self.env.timeout(ai_dur_min)
            task.ai_duration_min = ai_dur_min
            self.total_ai_busy_minutes += ai_dur_min

        # 4. Triage Decision: Referable or Routine
        is_ref = random.random() < self.referral_rate
        task.is_referable = is_ref

        # 5. Doctor Review (Only if referable)
        if is_ref:
            doc_queue_start = self.env.now
            with self.doctor_resource.request() as req:
                yield req
                task.doctor_wait_min = self.env.now - doc_queue_start
                
                doc_dur_min = random.uniform(
                    self.doctor_review_time_min * 0.8,
                    self.doctor_review_time_min * 1.2
                )
                yield self.env.timeout(doc_dur_min)
                task.doctor_duration_min = doc_dur_min
                self.total_doctor_busy_minutes += doc_dur_min

        task.total_turnaround_time_min = self.env.now - task.arrival_time_min
        self.completed_tasks.append(task)

    def phc_patient_generator(self, phc_index: int):
        """SimPy process generating arriving patients throughout the operating day at a given PHC."""
        total_patients = self.patients_per_phc_day
        if total_patients <= 0:
            return
            
        inter_arrival_mean = self.sim_duration_minutes / float(total_patients)
        
        for i in range(total_patients):
            task_id = f"TASK-PHC{phc_index+1}-{i+1:03d}"
            patient_id = f"PAT-RUR-{phc_index+1:02d}-{i+1:03d}"
            
            task = ScreeningTaskRecord(
                task_id=task_id,
                patient_id=patient_id,
                phc_id=f"PHC-{phc_index+1:02d}",
                arrival_time_min=self.env.now
            )
            self.env.process(self.patient_screening_lifecycle(task))
            
            # Wait for next patient arrival (exponential distribution)
            next_arrival_gap = random.expovariate(1.0 / max(inter_arrival_mean, 0.1))
            yield self.env.timeout(next_arrival_gap)

    def run_simulation(self) -> List[ScreeningTaskRecord]:
        """Execute the discrete-event simulation until the operational day finishes."""
        for phc_idx in range(self.num_phcs):
            self.env.process(self.phc_patient_generator(phc_idx))
            
        self.env.run(until=self.sim_duration_minutes)
        return self.completed_tasks
