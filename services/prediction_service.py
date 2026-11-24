import json
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd

from models.demand_predictor import DemandPredictor


class PredictionService:
    """Menghubungkan DemandPredictor, artefak evaluasi model, dan horizon input simulasi."""

    def __init__(
        self,
        predictor: DemandPredictor,
        metrics_path: Path,
        sim_horizon_path: Path,
        processed_data_path: Path,
    ):
        self.predictor = predictor
        self.metrics_path = Path(metrics_path)
        self.sim_horizon_path = Path(sim_horizon_path)
        self.processed_data_path = Path(processed_data_path)

    def get_model_performance(self) -> Dict[str, Any]:
        if not self.metrics_path.exists():
            return {
                "available": False,
                "message": (
                    "File artifacts/model_metrics.json belum ditemukan. "
                    "Jalankan 04_model_evaluation.ipynb terlebih dahulu."
                ),
            }

        with open(self.metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        # Sertakan sampel aktual vs prediksi dari test_predictions.csv jika tersedia
        preds_csv = self.metrics_path.parent / "test_predictions.csv"
        chart_series = {"timestamps": [], "actual": [], "predicted": []}
        if preds_csv.exists():
            df_p = pd.read_csv(preds_csv).head(120)
            best_name = metrics.get("best_model")
            chart_series["timestamps"] = df_p["datetime"].astype(str).tolist()
            chart_series["actual"] = df_p["actual"].round(2).tolist()
            if best_name in df_p.columns:
                chart_series["predicted"] = df_p[best_name].round(2).tolist()

        return {
            "available": True,
            "metrics": metrics,
            "chart_series": chart_series,
        }

    def predict_interactive(self, form_or_json: Dict[str, Any]) -> Dict[str, Any]:
        return self.predictor.predict_single(form_or_json)

    def get_simulation_demand_horizon(self, hours: int = 168) -> Tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Menyiapkan deret prediksi permintaan (dari model terlatih), kondisi cuaca,
        dan stempel waktu sepanjang `hours` jam untuk simulasi System Dynamics.
        """
        if self.sim_horizon_path.exists():
            df = pd.read_csv(self.sim_horizon_path)
        elif self.processed_data_path.exists():
            full_df = pd.read_csv(self.processed_data_path)
            split_idx = int(len(full_df) * 0.80)
            df = full_df.iloc[split_idx:].copy().reset_index(drop=True)
        else:
            raise FileNotFoundError("Dataset bersih maupun horizon simulasi tidak ditemukan.")

        if len(df) == 0:
            raise ValueError("Data horizon simulasi kosong.")

        # Jika permintaan durasi melebihi panjang frame, ulangi siklus secara periodik
        if hours <= len(df):
            sub_df = df.iloc[:hours].copy().reset_index(drop=True)
        else:
            repeats = (hours // len(df)) + 1
            sub_df = pd.concat([df] * repeats, ignore_index=True).iloc[:hours].copy()

        if self.predictor.is_ready() and all(
            col in sub_df.columns for col in self.predictor.feature_config.get("feature_order", [])
        ):
            demand_arr = self.predictor.predict_batch(sub_df)
        elif "predicted_demand" in sub_df.columns:
            demand_arr = np.maximum(sub_df["predicted_demand"].to_numpy(dtype=float), 0.0)
        else:
            self.predictor._ensure_ready()
            demand_arr = self.predictor.predict_batch(sub_df)

        weather_arr = sub_df["weather"].to_numpy(dtype=int) if "weather" in sub_df.columns else np.ones(hours, dtype=int)
        timestamps = sub_df["datetime"].astype(str).tolist() if "datetime" in sub_df.columns else [f"Jam-{i+1}" for i in range(hours)]

        return demand_arr, weather_arr, timestamps
