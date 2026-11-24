import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd


class DataService:
    """Mengelola pembacaan dataset mentah, dataset hasil preprocessing, dan ringkasan EDA."""

    def __init__(
        self,
        raw_path: Path,
        processed_path: Path,
        eda_summary_path: Path,
        prep_summary_path: Path,
    ):
        self.raw_path = Path(raw_path)
        self.processed_path = Path(processed_path)
        self.eda_summary_path = Path(eda_summary_path)
        self.prep_summary_path = Path(prep_summary_path)

    def get_dataset_overview(self, preview_rows: int = 15) -> Dict[str, Any]:
        raw_exists = self.raw_path.exists()
        processed_exists = self.processed_path.exists()

        eda_summary = {}
        if self.eda_summary_path.exists():
            with open(self.eda_summary_path, "r", encoding="utf-8") as f:
                eda_summary = json.load(f)

        prep_summary = {}
        if self.prep_summary_path.exists():
            with open(self.prep_summary_path, "r", encoding="utf-8") as f:
                prep_summary = json.load(f)

        info = {
            "source_name": "Kaggle Bike Sharing Demand (data/raw/train.csv)",
            "raw_exists": raw_exists,
            "processed_exists": processed_exists,
            "preprocessing_status": "Selesai (cleaned_data.csv tersedia)" if processed_exists else "Belum dijalankan",
            "eda_summary": eda_summary,
            "preprocessing_summary": prep_summary,
            "preview_columns": [],
            "preview_records": [],
            "descriptive_stats": [],
        }

        target_df = None
        if processed_exists:
            target_df = pd.read_csv(self.processed_path)
        elif raw_exists:
            target_df = pd.read_csv(self.raw_path)

        if target_df is not None:
            info["num_rows"] = int(target_df.shape[0])
            info["num_columns"] = int(target_df.shape[1])
            info["preview_columns"] = target_df.columns.tolist()
            info["preview_records"] = target_df.head(preview_rows).round(3).to_dict(orient="records")

            desc = target_df.describe().T.reset_index().rename(columns={"index": "feature"})
            info["descriptive_stats"] = desc.round(3).to_dict(orient="records")

        return info

    def get_demand_patterns(self) -> Dict[str, Any]:
        """Menghitung agregasi pola permintaan aktual per jam, cuaca, dan musim untuk grafik."""
        if self.processed_path.exists():
            df = pd.read_csv(self.processed_path)
        elif self.raw_path.exists():
            df = pd.read_csv(self.raw_path)
            df["hour"] = pd.to_datetime(df["datetime"]).dt.hour
        else:
            return {"available": False, "message": "Dataset belum tersedia."}

        hourly_work = (
            df[df["workingday"] == 1].groupby("hour")["count"].mean().reindex(range(24), fill_value=0).round(2).tolist()
        )
        hourly_weekend = (
            df[df["workingday"] == 0].groupby("hour")["count"].mean().reindex(range(24), fill_value=0).round(2).tolist()
        )
        hourly_overall = df.groupby("hour")["count"].mean().reindex(range(24), fill_value=0).round(2).tolist()

        weather_map = {
            1: "1: Cerah / Berawan Sebagian",
            2: "2: Berkabut / Mendung",
            3: "3: Hujan / Salju Ringan",
            4: "4: Hujan Lebat / Ekstrem",
        }
        weather_grp = df.groupby("weather")["count"].agg(["count", "mean", "median"]).reset_index()
        weather_data = [
            {
                "weather_code": int(row["weather"]),
                "label": weather_map.get(int(row["weather"]), f"Cuaca {int(row['weather'])}"),
                "samples": int(row["count"]),
                "mean_demand": round(float(row["mean"]), 2),
                "median_demand": round(float(row["median"]), 2),
            }
            for _, row in weather_grp.iterrows()
        ]

        season_map = {
            1: "1: Spring (Semi)",
            2: "2: Summer (Panas)",
            3: "3: Fall (Gugur)",
            4: "4: Winter (Dingin)",
        }
        season_grp = df.groupby("season")["count"].agg(["count", "mean", "median"]).reset_index()
        season_data = [
            {
                "season_code": int(row["season"]),
                "label": season_map.get(int(row["season"]), f"Musim {int(row['season'])}"),
                "samples": int(row["count"]),
                "mean_demand": round(float(row["mean"]), 2),
                "median_demand": round(float(row["median"]), 2),
            }
            for _, row in season_grp.iterrows()
        ]

        return {
            "available": True,
            "hours": list(range(24)),
            "hourly_overall": hourly_overall,
            "hourly_workday": hourly_work,
            "hourly_weekend": hourly_weekend,
            "weather_patterns": weather_data,
            "season_patterns": season_data,
        }
