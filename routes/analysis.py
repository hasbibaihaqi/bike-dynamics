from flask import Blueprint, current_app, jsonify, request
from models.demand_predictor import ModelNotTrainedError
from models.validation import ValidationError
from routes import render_or_fallback

analysis_bp = Blueprint("analysis", __name__)


@analysis_bp.route("/demand-analysis", methods=["GET", "POST"])
def demand_analysis():
    services = current_app.extensions["services"]
    patterns = services["data"].get_demand_patterns()
    model_status = services["predictor"].get_status()

    prediction_result = None
    prediction_error = None
    input_values = {
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

    if request.method == "POST":
        raw_data = request.get_json(silent=True) or request.form.to_dict()
        input_values.update(raw_data)
        try:
            prediction_result = services["prediction"].predict_interactive(raw_data)
        except (ValidationError, ModelNotTrainedError) as exc:
            prediction_error = str(exc)

    payload = {
        "active_menu": "demand_analysis",
        "patterns": patterns,
        "model_status": model_status,
        "input_values": input_values,
        "prediction_result": prediction_result,
        "prediction_error": prediction_error,
    }
    return render_or_fallback("demand_analysis.html", "Demand Analysis", payload)


@analysis_bp.route("/api/predict-demand", methods=["POST"])
def api_predict_demand():
    services = current_app.extensions["services"]
    raw_data = request.get_json(silent=True) or request.form.to_dict()
    try:
        result = services["prediction"].predict_interactive(raw_data)
        return jsonify({"status": "success", "data": result})
    except ValidationError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    except ModelNotTrainedError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 503


@analysis_bp.route("/model-performance")
def model_performance():
    services = current_app.extensions["services"]
    perf_data = services["prediction"].get_model_performance()
    payload = {
        "active_menu": "model_performance",
        "performance": perf_data,
        "model_status": services["predictor"].get_status(),
    }
    return render_or_fallback("model_performance.html", "Model Performance", payload)
