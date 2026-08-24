"""
Unit Tests for Phase 6: District Dashboard Analytics & Operational Telemetry
"""

import pytest
from app.pages.dashboard import get_district_demo_data


def test_get_district_demo_data_default():
    """Verify district dashboard demo dataset structure and KPI metrics."""
    data = get_district_demo_data(timeframe="Last 14 Days", phc_filter="All PHCs")
    
    assert "total_screened" in data
    assert "gradable_rate" in data
    assert "referable_count" in data
    assert "urgent_count" in data
    assert "recaptures" in data
    assert "phc_data" in data
    
    assert data["total_screened"] > 0
    assert data["referable_count"] > 0
    assert data["gradable_rate"] > 90.0
    assert len(data["phc_data"]) >= 5


def test_get_district_demo_data_phc_filter():
    """Verify that filtering by a specific PHC centre isolates that centre's volume."""
    filtered_data = get_district_demo_data(timeframe="Last 14 Days", phc_filter="Meenangadi CHC")
    
    assert len(filtered_data["phc_data"]) == 1
    assert filtered_data["phc_data"][0]["phc"] == "Meenangadi CHC"
    assert filtered_data["total_screened"] == filtered_data["phc_data"][0]["patients"]


def test_get_district_demo_data_timeframes():
    """Verify that different timeframe horizons adjust volumes accordingly."""
    data_7d = get_district_demo_data(timeframe="Last 7 Days")
    data_30d = get_district_demo_data(timeframe="Last 30 Days")
    data_ytd = get_district_demo_data(timeframe="Year-to-Date (YTD)")
    
    assert data_7d["total_screened"] < data_30d["total_screened"] < data_ytd["total_screened"]
