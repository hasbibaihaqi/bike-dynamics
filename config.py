import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class Config:
    APP_NAME = "BIKE-DYNAMICS"
    APP_TAGLINE = "Dynamic System Simulation for Bike Sharing Management"
    RESEARCH_TITLE = (
        "Pemodelan dan Simulasi Sistem Dinamik untuk Optimalisasi Ketersediaan "
        "Armada Sepeda pada Layanan Bike Sharing Berdasarkan Fluktuasi Permintaan "
        "dan Kondisi Cuaca"
    )
    SECRET_KEY = os.environ.get("SECRET_KEY", "bike-dynamics-uts-mosi-2026")

    # Direktori utama proyek
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_PATH = DATA_DIR / "raw" / "train.csv"
    PROCESSED_DATA_PATH = DATA_DIR / "processed" / "cleaned_data.csv"

    ARTIFACTS_DIR = BASE_DIR / "artifacts"
    MODEL_PATH = ARTIFACTS_DIR / "demand_model.joblib"
    PIPELINE_PATH = ARTIFACTS_DIR / "preprocessing_pipeline.joblib"
    METRICS_PATH = ARTIFACTS_DIR / "model_metrics.json"
    FEATURE_CONFIG_PATH = ARTIFACTS_DIR / "feature_config.json"
    EDA_SUMMARY_PATH = ARTIFACTS_DIR / "eda_summary.json"
    PREP_SUMMARY_PATH = ARTIFACTS_DIR / "preprocessing_summary.json"
    SD_ASSUMPTIONS_PATH = ARTIFACTS_DIR / "sd_assumptions.json"
    SIM_HORIZON_PATH = ARTIFACTS_DIR / "simulation_horizon_input.csv"
    SCENARIO_JSON_PATH = ARTIFACTS_DIR / "scenario_experiments.json"

    INSTANCE_DIR = BASE_DIR / "instance"
    SQLITE_DB_PATH = INSTANCE_DIR / "bike_dynamics.db"
    TEMPLATES_DIR = BASE_DIR / "templates"
    STATIC_DIR = BASE_DIR / "static"

    # Default parameter System Dynamics (konsisten dengan 05_system_dynamics.ipynb)
    DEFAULT_SD_PARAMS = {
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
