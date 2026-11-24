import io
import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd

from models.scenarios import ScenarioRunner
from models.system_dynamics import SystemDynamicsSimulator
from services.prediction_service import PredictionService
from services.simulation_service import SimulationService


class AnalysisService:
    """
    Menangani Scenario Analysis (4 skenario), Sensitivity Analysis,
    serta penyusunan laporan & ekspor CSV untuk Results and Reports.
    """

    def __init__(
        self,
        simulator: SystemDynamicsSimulator,
        scenario_runner: ScenarioRunner,
        prediction_service: PredictionService,
        simulation_service: SimulationService,
        scenario_json_path: Path,
    ):
        self.simulator = simulator
        self.scenario_runner = scenario_runner
        self.prediction_service = prediction_service
        self.simulation_service = simulation_service
        self.scenario_json_path = Path(scenario_json_path)

    def run_scenario_comparison(self, hours: int = 168) -> Dict[str, Any]:
        base_params = self.simulation_service.get_assumptions()
        demand_arr, weather_arr, timestamps = self.prediction_service.get_simulation_demand_horizon(hours=hours)
        live_result = self.scenario_runner.run_all_scenarios(
            demand_series=demand_arr,
            weather_series=weather_arr,
            timestamps=timestamps,
            base_params=base_params,
        )

        notebook_reference = {}
        if self.scenario_json_path.exists():
            with open(self.scenario_json_path, "r", encoding="utf-8") as f:
                notebook_reference = json.load(f)

        return {
            "horizon_hours": hours,
            "best_scenario": live_result["best_scenario"],
            "scenarios": live_result["scenarios"],
            "notebook_artifact_available": bool(notebook_reference),
        }

    def run_sensitivity_analysis(
        self,
        parameter_name: str = "damage_rate",
        hours: int = 168,
    ) -> Dict[str, Any]:
        grids = {
            "damage_rate": {
                "label": "Tingkat Kerusakan (Damage Rate)",
                "unit": "Fraksi",
                "values": [0.01, 0.02, 0.04, 0.06, 0.08, 0.12],
            },
            "repair_duration_hours": {
                "label": "Durasi Perbaikan (Repair Duration)",
                "unit": "Jam",
                "values": [2.0, 3.0, 4.0, 6.0, 10.0, 16.0],
            },
            "rental_duration_hours": {
                "label": "Durasi Sewa (Rental Duration)",
                "unit": "Jam",
                "values": [1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
            },
            "total_fleet": {
                "label": "Kapasitas Total Armada (Fleet Capacity)",
                "unit": "Sepeda",
                "values": [450.0, 525.0, 600.0, 675.0, 750.0, 825.0],
            },
            "demand_multiplier": {
                "label": "Perubahan Permintaan (Demand Multiplier)",
                "unit": "Pengali",
                "values": [0.8, 0.9, 1.0, 1.1, 1.2, 1.35],
            },
        }

        if parameter_name not in grids:
            parameter_name = "damage_rate"

        selected = grids[parameter_name]
        base_params = self.simulation_service.get_assumptions()
        demand_arr, weather_arr, timestamps = self.prediction_service.get_simulation_demand_horizon(hours=hours)

        sweep_results = []
        for val in selected["values"]:
            sim_kwargs = {
                "total_fleet": float(base_params["total_fleet"]),
                "initial_available": float(base_params["initial_available"]),
                "initial_rented": float(base_params["initial_rented"]),
                "initial_maintenance": float(base_params["initial_maintenance"]),
                "rental_duration_hours": float(base_params["rental_duration_hours"]),
                "damage_rate": float(base_params["damage_rate"]),
                "repair_duration_hours": float(base_params["repair_duration_hours"]),
                "demand_multiplier": 1.0,
            }
            sim_kwargs[parameter_name] = float(val)
            if parameter_name == "total_fleet":
                sim_kwargs["initial_available"] = round(float(val) * 0.85, 2)
                sim_kwargs["initial_rented"] = round(float(val) * 0.10, 2)
                sim_kwargs["initial_maintenance"] = round(float(val) * 0.05, 2)

            out = self.simulator.simulate(
                demand_series=demand_arr,
                weather_series=weather_arr,
                timestamps=timestamps,
                **sim_kwargs,
            )
            s = out["summary"]
            sweep_results.append(
                {
                    "parameter_value": val,
                    "total_demand": s["total_demand"],
                    "total_rentals": s["total_rentals"],
                    "total_unmet_demand": s["total_unmet_demand"],
                    "service_level_pct": s["service_level_pct"],
                    "avg_available_bikes": s["avg_available_bikes"],
                    "avg_utilization_pct": s["avg_utilization_pct"],
                    "avg_maintenance_bikes": s["avg_maintenance_bikes"],
                }
            )

        return {
            "parameter_name": parameter_name,
            "parameter_label": selected["label"],
            "unit": selected["unit"],
            "available_parameters": {k: v["label"] for k, v in grids.items()},
            "results": sweep_results,
        }

    def export_simulation_csv(self, run_id: int | None = None) -> tuple[str, str]:
        run_data = (
            self.simulation_service.get_run_by_id(run_id)
            if run_id is not None
            else self.simulation_service.get_latest_or_baseline_run()
        )
        if run_data is None:
            run_data = self.simulation_service.get_latest_or_baseline_run()

        df = pd.DataFrame(run_data["timeseries"])
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        filename = f"bike_dynamics_simulation_run_{run_data['run_id']}.csv"
        return buf.getvalue(), filename
