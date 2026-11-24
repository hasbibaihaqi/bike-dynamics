import json
from pathlib import Path
from typing import Any, Dict, List
import joblib
import numpy as np
import pandas as pd

from models.validation import enrich_and_validate_demand_features, ValidationError


class ModelNotTrainedError(RuntimeError):
    """Dilemparkan ketika file artefak model dari notebook belum tersedia."""


class DemandPredictor:
    """
    Bertanggung jawab untuk:
    1. Memuat model terlatih (.joblib) dari folder artifacts/.
    2. Memuat preprocessing pipeline (.joblib) dari folder artifacts/.
    3. Memeriksa ketersediaan file model tanpa melakukan training ulang di Flask.
    4. Melakukan prediksi permintaan (tunggal & batch).
    5. Menangani kesalahan input.
    6. Memastikan urutan fitur sesuai dengan konfigurasi saat training.
    """

    def __init__(self, model_path: Path, pipeline_path: Path, feature_config_path: Path):
        self.model_path = Path(model_path)
        self.pipeline_path = Path(pipeline_path)
        self.feature_config_path = Path(feature_config_path)

        self.model = None
        self.pipeline = None
        self.feature_config: Dict[str, Any] = {}
        self.load_error: str | None = None

        self.load_artifacts()

    def load_artifacts(self) -> bool:
        missing = []
        if not self.model_path.exists():
            missing.append(str(self.model_path.name))
        if not self.pipeline_path.exists():
            missing.append(str(self.pipeline_path.name))
        if not self.feature_config_path.exists():
            missing.append(str(self.feature_config_path.name))

        if missing:
            self.model = None
            self.pipeline = None
            self.load_error = (
                f"File model terlatih belum tersedia ({', '.join(missing)}). "
                "Silakan lakukan proses training dan evaluasi terlebih dahulu melalui "
                "Jupyter Notebook (03_train_demand_model.ipynb dan 04_model_evaluation.ipynb)."
            )
            return False

        try:
            self.model = joblib.load(self.model_path)
            self.pipeline = joblib.load(self.pipeline_path)
            with open(self.feature_config_path, "r", encoding="utf-8") as f:
                self.feature_config = json.load(f)
            self.load_error = None
            return True
        except Exception as exc:
            self.model = None
            self.pipeline = None
            self.load_error = f"Gagal memuat artefak model terlatih: {exc}"
            return False

    def is_ready(self) -> bool:
        return self.model is not None and self.pipeline is not None and bool(self.feature_config)

    def get_status(self) -> Dict[str, Any]:
        return {
            "ready": self.is_ready(),
            "selected_model": self.feature_config.get("selected_model_name", "Belum tersedia"),
            "feature_count": len(self.feature_config.get("feature_order", [])),
            "message": "Model terlatih siap digunakan untuk inferensi." if self.is_ready() else self.load_error,
        }

    def _ensure_ready(self) -> None:
        if not self.is_ready():
            self.load_artifacts()
        if not self.is_ready():
            raise ModelNotTrainedError(self.load_error or "Model belum dilatih.")

    def predict_single(self, input_features: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_ready()
        feature_order: List[str] = self.feature_config["feature_order"]
        defaults: Dict[str, Any] = self.feature_config.get("feature_defaults", {})

        df_features = enrich_and_validate_demand_features(input_features, feature_order, defaults)
        X_transformed = self.pipeline.transform(df_features[feature_order])
        raw_pred = float(self.model.predict(X_transformed)[0])
        predicted_count = max(0.0, raw_pred)

        return {
            "predicted_demand": round(predicted_count, 2),
            "predicted_demand_int": int(round(predicted_count)),
            "model_used": self.feature_config.get("selected_model_name", "Best Demand Model"),
            "validated_features": df_features.iloc[0].to_dict(),
        }

    def predict_batch(self, df_features: pd.DataFrame) -> np.ndarray:
        self._ensure_ready()
        feature_order: List[str] = self.feature_config["feature_order"]

        missing_cols = [col for col in feature_order if col not in df_features.columns]
        if missing_cols:
            raise ValidationError(f"Kolom fitur tidak lengkap untuk prediksi batch: {missing_cols}")

        ordered_df = df_features[feature_order].copy()
        X_transformed = self.pipeline.transform(ordered_df)
        preds = np.asarray(self.model.predict(X_transformed), dtype=float)
        return np.maximum(preds, 0.0)
