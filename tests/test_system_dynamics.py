import numpy as np
import pytest

from models.scenarios import ScenarioRunner
from models.system_dynamics import SystemDynamicsSimulator
from models.validation import ValidationError, validate_simulation_params


def test_fleet_conservation_and_non_negativity():
    simulator = SystemDynamicsSimulator()
    rng = np.random.default_rng(42)
    demand_series = rng.uniform(50.0, 950.0, size=72)
    weather_series = rng.choice([1, 2, 3, 4], size=72)

    result = simulator.simulate(
        demand_series=demand_series,
        weather_series=weather_series,
        total_fleet=500.0,
        initial_available=400.0,
        initial_rented=70.0,
        initial_maintenance=30.0,
        rental_duration_hours=1.5,
        damage_rate=0.05,
        repair_duration_hours=6.0,
        demand_multiplier=1.0,
    )

    summary = result["summary"]
    assert summary["conservation_verified"] is True
    assert summary["max_conservation_error"] < 1e-5

    for row in result["timeseries"]:
        assert row["available_bikes"] >= 0.0
        assert row["rented_bikes"] >= 0.0
        assert row["maintenance_bikes"] >= 0.0
        total_step = row["available_bikes"] + row["rented_bikes"] + row["maintenance_bikes"]
        assert abs(row['total_fleet'] - 500.0) < 1e-4
        assert abs(total_step - 500.0) <= 0.02


def test_stockout_generates_unmet_demand():
    simulator = SystemDynamicsSimulator()
    result = simulator.simulate(
        demand_series=[150.0, 150.0, 150.0],
        weather_series=[1, 1, 1],
        total_fleet=100.0,
        initial_available=40.0,
        initial_rented=50.0,
        initial_maintenance=10.0,
        rental_duration_hours=2.0,
        damage_rate=0.04,
        repair_duration_hours=6.0,
    )
    assert result["summary"]["total_unmet_demand"] > 0.0
    assert result["summary"]["service_level_pct"] < 100.0


def test_scenario_runner_five_policies():
    runner = ScenarioRunner()
    demand_series = [180.0, 320.0, 540.0, 620.0, 410.0, 220.0] * 4
    weather_series = [1, 1, 2, 3, 1, 1] * 4
    base_params = {
        "total_fleet": 600.0,
        "rental_duration_hours": 1.5,
        "damage_rate": 0.04,
        "repair_duration_hours": 6.0,
    }

    output = runner.run_all_scenarios(
        demand_series=demand_series,
        weather_series=weather_series,
        timestamps=None,
        base_params=base_params,
    )

    assert len(output["scenarios"]) == 5
    best = output["best_scenario"]
    assert "scenario_id" in best
    assert 0.0 <= best["composite_score"] <= 100.0
    assert 0.0 <= best["service_level_pct"] <= 100.0


def test_simulation_param_validation():
    defaults = {
        "total_fleet": 600.0,
        "initial_available": 510.0,
        "initial_rented": 60.0,
        "initial_maintenance": 30.0,
        "rental_duration_hours": 1.5,
        "damage_rate": 0.04,
        "repair_duration_hours": 6.0,
        "demand_multiplier": 1.0,
        "simulation_hours": 168,
    }

    with pytest.raises(ValidationError):
        validate_simulation_params(
            {"total_fleet": 500.0, "initial_available": 480.0, "initial_maintenance": 50.0},
            defaults,
        )

    with pytest.raises(ValidationError):
        validate_simulation_params({"damage_rate": 1.5}, defaults)
