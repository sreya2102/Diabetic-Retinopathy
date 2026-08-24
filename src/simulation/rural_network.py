"""
Rural Tele-Network & Health Centre (PHC) Modeling (Phase 7)
Defines network constraints, image transmission rates, bandwidth throttling, and SimPy network channels.
"""

from dataclasses import dataclass
from typing import List, Optional
import random
import simpy


@dataclass
class PHCNode:
    """Represents a rural Primary Health Centre with local acquisition and connectivity constraints."""
    id: str
    name: str
    district: str
    patients_per_day: int = 25
    average_image_size_mb: float = 3.5
    network_bandwidth_mbps: float = 2.0  # Typical 2G/3G/Rural 4G uplink
    network_reliability: float = 0.92     # Transmission success probability
    offline_buffer_capacity: int = 500   # Max images stored locally if network drops


class RuralNetworkSimulator:
    """
    SimPy-integrated network simulation engine for rural tele-ophthalmology data routing.
    Models bandwidth contention, packet retransmission, and upload latency.
    """
    
    def __init__(self, env: Optional[simpy.Environment] = None, phcs: Optional[List[PHCNode]] = None):
        self.env = env
        self.phcs = phcs or []

    def calculate_transmission_time_seconds(
        self,
        image_size_mb: float,
        bandwidth_mbps: float,
        packet_loss_rate: float = 0.05
    ) -> float:
        """
        Compute image upload duration in seconds under constrained uplink bandwidth.
        Time = (Size in Megabits / Effective Bandwidth) * (1 + packet_loss_rate)
        """
        if bandwidth_mbps <= 0:
            return 999999.0
        size_megabits = image_size_mb * 8.0
        effective_bandwidth = max(bandwidth_mbps * (1.0 - packet_loss_rate), 0.05)
        base_time = size_megabits / effective_bandwidth
        # Add slight random jitter (5-15%)
        jitter = random.uniform(0.95, 1.15)
        return float(base_time * jitter)

    def calculate_transmission_time(
        self,
        image_size_mb: float,
        bandwidth_mbps: float,
        packet_loss_rate: float = 0.05
    ) -> float:
        """Alias for calculate_transmission_time_seconds."""
        return self.calculate_transmission_time_seconds(image_size_mb, bandwidth_mbps, packet_loss_rate)

    def transmit_image_process(
        self,
        env: simpy.Environment,
        task_id: str,
        image_size_mb: float,
        bandwidth_mbps: float
    ):
        """SimPy process modeling the upload delay of a fundus photograph."""
        tx_time_sec = self.calculate_transmission_time_seconds(image_size_mb, bandwidth_mbps)
        tx_time_min = tx_time_sec / 60.0  # Convert to simulation minutes
        yield env.timeout(tx_time_min)
