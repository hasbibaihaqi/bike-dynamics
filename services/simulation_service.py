import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from models.system_dynamics import SystemDynamicsSimulator
from models.validation import validate_simulation_params
from services.prediction_service import PredictionService


class SimulationService:
    """
    Menjalankan simulasi System Dynamics berdasarkan parameter Simulation Center
    dan menyimpan riwayat simulasi ke database SQLite (instance/bike_dynamics.db).
    """

    def __init__(
        self,
        simulator: SystemDynamicsSimulator,
        prediction_service: PredictionService,
        assumptions_path: Path,
        db_path: Path,
        default_params: Dict[str, Any],
    ):
        self.simulator = simulator
        self.prediction_service = prediction_service
        self.assumptions_path = Path(assumptions_path)
        self.db_path = Path(db_path)
        self.default_params = dict(default_params)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS simulation_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    run_label TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    timeseries_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def get_assumptions(self) -> Dict[str, Any]:
        if self.assumptions_path.exists():
            with open(self.assumptions_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            return {**self.default_params, **loaded}
        return dict(self.default_params)

    def run_custom_simulation(
        self,
        raw_params: Dict[str, Any],
        run_label: str = "Simulasi Kustom",
        save_to_db: bool = True,
    ) -> Dict[str, Any]:
        clean_params = validate_simulation_params(raw_params, self.get_assumptions())
        hours = clean_params["simulation_hours"]

        demand_arr, weather_arr, timestamps = self.prediction_service.get_simulation_demand_horizon(hours=hours)

        sim_result = self.simulator.simulate(
            demand_series=demand_arr,
            weather_series=weather_arr,
            timestamps=timestamps,
            total_fleet=clean_params["total_fleet"],
            initial_available=clean_params["initial_available"],
            initial_rented=clean_params["initial_rented"],
            initial_maintenance=clean_params["initial_maintenance"],
            rental_duration_hours=clean_params["rental_duration_hours"],
            damage_rate=clean_params["damage_rate"],
            repair_duration_hours=clean_params["repair_duration_hours"],
            demand_multiplier=clean_params["demand_multiplier"],
        )

        run_id = None
        created_at = datetime.now().isoformat(timespec="seconds")
        if save_to_db:
            with self._get_conn() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO simulation_runs (created_at, run_label, params_json, summary_json, timeseries_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        created_at,
                        run_label,
                        json.dumps(clean_params),
                        json.dumps(sim_result["summary"]),
                        json.dumps(sim_result["timeseries"]),
                    ),
                )
                run_id = cur.lastrowid
                conn.commit()

        return {
            "run_id": run_id,
            "created_at": created_at,
            "run_label": run_label,
            "parameters": clean_params,
            "summary": sim_result["summary"],
            "timeseries": sim_result["timeseries"],
        }

    def get_latest_or_baseline_run(self) -> Dict[str, Any]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM simulation_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()

        if row is not None:
            return {
                "run_id": row["id"],
                "created_at": row["created_at"],
                "run_label": row["run_label"],
                "parameters": json.loads(row["params_json"]),
                "summary": json.loads(row["summary_json"]),
                "timeseries": json.loads(row["timeseries_json"]),
            }

        return self.run_custom_simulation(
            raw_params=self.get_assumptions(),
            run_label="Baseline Otomatis (Normal Condition)",
            save_to_db=True,
        )

    def get_run_by_id(self, run_id: int) -> Dict[str, Any] | None:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM simulation_runs WHERE id = ?", (int(run_id),)
            ).fetchone()
        if row is None:
            return None
        return {
            "run_id": row["id"],
            "created_at": row["created_at"],
            "run_label": row["run_label"],
            "parameters": json.loads(row["params_json"]),
            "summary": json.loads(row["summary_json"]),
            "timeseries": json.loads(row["timeseries_json"]),
        }

    def list_recent_runs(self, limit: int = 15) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT id, created_at, run_label, params_json, summary_json FROM simulation_runs ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()

        history = []
        for r in rows:
            history.append(
                {
                    "run_id": r["id"],
                    "created_at": r["created_at"],
                    "run_label": r["run_label"],
                    "parameters": json.loads(r["params_json"]),
                    "summary": json.loads(r["summary_json"]),
                }
            )
        return history
