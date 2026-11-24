from typing import Any, Dict, List, Sequence
from models.system_dynamics import SystemDynamicsSimulator


class ScenarioRunner:
    """
    Menjalankan lima skenario kebijakan pengelolaan armada dan menentukan
    skenario terbaik berdasarkan skor efektivitas komposit.
    """

    def __init__(self, simulator: SystemDynamicsSimulator | None = None):
        self.simulator = simulator or SystemDynamicsSimulator()

    @staticmethod
    def build_default_scenarios(base_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        base_fleet = float(base_params.get("total_fleet", 600.0))
        expanded_fleet = round(base_fleet * 1.20, 2)
        rental_dur = float(base_params.get("rental_duration_hours", 1.5))
        damage_rt = float(base_params.get("damage_rate", 0.04))
        repair_dur = float(base_params.get("repair_duration_hours", 6.0))

        return [
            {
                "scenario_id": "scenario_1_normal",
                "scenario_name": "Skenario 1: Normal Condition",
                "short_label": "Normal (Baseline)",
                "description": "Kondisi operasional standar sesuai asumsi dasar tanpa intervensi kebijakan tambahan.",
                "params": {
                    "total_fleet": base_fleet,
                    "initial_available": round(base_fleet * 0.85, 2),
                    "initial_rented": round(base_fleet * 0.10, 2),
                    "initial_maintenance": round(base_fleet * 0.05, 2),
                    "rental_duration_hours": rental_dur,
                    "damage_rate": damage_rt,
                    "repair_duration_hours": repair_dur,
                    "demand_multiplier": 1.0,
                },
            },
            {
                "scenario_id": "scenario_2_high_demand",
                "scenario_name": "Skenario 2: High Demand (+20%)",
                "short_label": "High Demand (+20%)",
                "description": "Lonjakan permintaan penyewaan sebesar +20% pada kapasitas armada eksisting.",
                "params": {
                    "total_fleet": base_fleet,
                    "initial_available": round(base_fleet * 0.85, 2),
                    "initial_rented": round(base_fleet * 0.10, 2),
                    "initial_maintenance": round(base_fleet * 0.05, 2),
                    "rental_duration_hours": rental_dur,
                    "damage_rate": damage_rt,
                    "repair_duration_hours": repair_dur,
                    "demand_multiplier": 1.20,
                },
            },
            {
                "scenario_id": "scenario_3_fleet_expansion",
                "scenario_name": "Skenario 3: Fleet Expansion (+20%)",
                "short_label": "Fleet Expansion (+20%)",
                "description": "Penambahan kapasitas armada sebesar +20% untuk meredam permintaan tidak terpenuhi.",
                "params": {
                    "total_fleet": expanded_fleet,
                    "initial_available": round(expanded_fleet * 0.85, 2),
                    "initial_rented": round(expanded_fleet * 0.10, 2),
                    "initial_maintenance": round(expanded_fleet * 0.05, 2),
                    "rental_duration_hours": rental_dur,
                    "damage_rate": damage_rt,
                    "repair_duration_hours": repair_dur,
                    "demand_multiplier": 1.0,
                },
            },
            {
                "scenario_id": "scenario_4_maintenance_opt",
                "scenario_name": "Skenario 4: Maintenance Optimization",
                "short_label": "Maintenance Opt.",
                "description": "Percepatan durasi perbaikan bengkel sebesar 50% (dari 6.0 jam menjadi 3.0 jam).",
                "params": {
                    "total_fleet": base_fleet,
                    "initial_available": round(base_fleet * 0.85, 2),
                    "initial_rented": round(base_fleet * 0.10, 2),
                    "initial_maintenance": round(base_fleet * 0.05, 2),
                    "rental_duration_hours": rental_dur,
                    "damage_rate": damage_rt,
                    "repair_duration_hours": round(max(1.0, repair_dur * 0.5), 2),
                    "demand_multiplier": 1.0,
                },
            },
            {
                "scenario_id": "scenario_5_high_demand_fleet_expansion",
                "scenario_name": "Skenario 5: High Demand + Fleet Expansion",
                "short_label": "High Demand + Fleet Exp.",
                "description": "Kombinasi peningkatan permintaan (+20%) dan penambahan kapasitas armada (+20%) untuk menguji daya serap ekspansi armada.",
                "params": {
                    "total_fleet": expanded_fleet,
                    "initial_available": round(expanded_fleet * 0.85, 2),
                    "initial_rented": round(expanded_fleet * 0.10, 2),
                    "initial_maintenance": round(expanded_fleet * 0.05, 2),
                    "rental_duration_hours": rental_dur,
                    "damage_rate": damage_rt,
                    "repair_duration_hours": repair_dur,
                    "demand_multiplier": 1.20,
                },
            },
        ]

    def run_all_scenarios(
        self,
        demand_series: Sequence[float],
        weather_series: Sequence[int] | None,
        timestamps: Sequence[str] | None,
        base_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        scenarios_def = self.build_default_scenarios(base_params)
        results = []

        for sc in scenarios_def:
            sim_out = self.simulator.simulate(
                demand_series=demand_series,
                weather_series=weather_series,
                timestamps=timestamps,
                **sc["params"],
            )
            summary = sim_out["summary"]
            results.append(
                {
                    "scenario_id": sc["scenario_id"],
                    "scenario_name": sc["scenario_name"],
                    "short_label": sc["short_label"],
                    "description": sc["description"],
                    "parameters": sc["params"],
                    "summary": summary,
                    "timeseries": sim_out["timeseries"],
                }
            )

        # Hitung Composite Policy Effectiveness Score (konsisten dengan 06_scenario_experiments.ipynb)
        max_rentals_per_bike = max(
            (r["summary"]["total_rentals"] / r["parameters"]["total_fleet"]) for r in results
        ) or 1.0

        for r in results:
            s = r["summary"]
            fleet = r["parameters"]["total_fleet"]
            score_service = s["service_level_pct"]
            score_health = (1.0 - (s["avg_maintenance_bikes"] / fleet)) * 100.0
            score_prod = ((s["total_rentals"] / fleet) / max_rentals_per_bike) * 100.0
            composite = 0.50 * score_service + 0.25 * score_health + 0.25 * score_prod
            s["composite_score"] = round(composite, 2)

        ranked = sorted(results, key=lambda x: (x["summary"]["composite_score"], x["summary"]["service_level_pct"]), reverse=True)
        best = ranked[0]

        return {
            "scenarios": results,
            "best_scenario": {
                "scenario_id": best["scenario_id"],
                "scenario_name": best["scenario_name"],
                "composite_score": best["summary"]["composite_score"],
                "service_level_pct": best["summary"]["service_level_pct"],
                "total_unmet_demand": best["summary"]["total_unmet_demand"],
                "rationale": (
                    f"{best['scenario_name']} memperoleh skor efektivitas tertinggi "
                    f"({best['summary']['composite_score']}/100) dengan tingkat pemenuhan "
                    f"permintaan {best['summary']['service_level_pct']}% serta efisiensi "
                    f"ketersediaan dan perawatan armada yang seimbang."
                ),
            },
        }
