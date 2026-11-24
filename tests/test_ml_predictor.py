import pytest

from config import Config
from models.demand_predictor import DemandPredictor, ModelNotTrainedError
from models.validation import ValidationError


def test_artifacts_loaded_and_ready():
    predictor = DemandPredictor(
        model_path=Config.MODEL_PATH,
        pipeline_path=Config.PIPELINE_PATH,
        feature_config_path=Config.FEATURE_CONFIG_PATH,
    )
    status = predictor.get_status()
    assert predictor.is_ready() is True
    assert status["ready"] is True
    assert status["feature_count"] > 0


def test_predict_single_valid_input():
    predictor = DemandPredictor(
        model_path=Config.MODEL_PATH,
        pipeline_path=Config.PIPELINE_PATH,
        feature_config_path=Config.FEATURE_CONFIG_PATH,
    )
    sample = {
        "season": 2,
        "weather": 1,
        "year": 2012,
        "month": 6,
        "day": 15,
        "hour": 8,
        "dayofweek": 2,
        "holiday": 0,
        "workingday": 1,
        "temp": 24.0,
        "atemp": 26.0,
        "humidity": 55.0,
        "windspeed": 12.0,
    }
    res = predictor.predict_single(sample)
    assert res["predicted_demand"] >= 0.0
    assert res["validated_features"]["is_rush_hour"] == 1
    assert isinstance(res["model_used"], str) and len(res["model_used"]) > 0


def test_predict_single_invalid_input_raises_validation_error():
    predictor = DemandPredictor(
        model_path=Config.MODEL_PATH,
        pipeline_path=Config.PIPELINE_PATH,
        feature_config_path=Config.FEATURE_CONFIG_PATH,
    )
    with pytest.raises(ValidationError):
        predictor.predict_single({"hour": 28, "weather": 1})

    with pytest.raises(ValidationError):
        predictor.predict_single({"hour": 10, "weather": 7})


def test_missing_artifact_raises_model_not_trained(tmp_path):
    predictor = DemandPredictor(
        model_path=tmp_path / "missing_model.joblib",
        pipeline_path=tmp_path / "missing_pipeline.joblib",
        feature_config_path=tmp_path / "missing_config.json",
    )
    assert predictor.is_ready() is False
    with pytest.raises(ModelNotTrainedError):
        predictor.predict_single({"hour": 8})
