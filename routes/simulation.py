from flask import Blueprint, current_app, jsonify, request
from models.demand_predictor import ModelNotTrainedError
from models.system_dynamics import SystemDynamicsSimulator
from models.validation import ValidationError
from routes import render_or_fallback

simulation_bp = Blueprint("simulation", __name__)


@simulation_bp.route("/system-dynamics")
def system_dynamics_overview():
    services = current_app.extensions["services"]
    assumptions = services["simulation"].get_assumptions()
    doc = SystemDynamicsSimulator.get_model_documentation(assumptions)
    payload = {
        "active_menu": "system_dynamics",
        "model_doc": doc,
    }
    return render_or_fallback("system_dynamics.html", "System Dynamics", payload)


@simulation_bp.route("/simulation-center", methods=["GET", "POST"])
def simulation_center():
    services = current_app.extensions["services"]
    assumptions = services["simulation"].get_assumptions()
    error_message = None

    if request.method == "POST":
        raw_data = request.get_json(silent=True) or request.form.to_dict()
        run_label = raw_data.pop("run_label", "").strip() or "Simulasi Kustom Pengguna"
        try:
            sim_run = services["simulation"].run_custom_simulation(
                raw_params=raw_data,
                run_label=run_label,
                save_to_db=True,
            )
        except (ValidationError, ModelNotTrainedError, FileNotFoundError) as exc:
            error_message = str(exc)
            sim_run = services["simulation"].get_latest_or_baseline_run()
    else:
        sim_run = services["simulation"].get_latest_or_baseline_run()

    payload = {
        "active_menu": "simulation_center",
        "assumptions": assumptions,
        "simulation_run": sim_run,
        "error_message": error_message,
    }
    return render_or_fallback("simulation_center.html", "Simulation Center", payload)


@simulation_bp.route("/api/simulate", methods=["POST"])
def api_run_simulation():
    services = current_app.extensions["services"]
    raw_data = request.get_json(silent=True) or request.form.to_dict()
    run_label = raw_data.pop("run_label", "").strip() or "API Simulation Run"
    try:
        sim_run = services["simulation"].run_custom_simulation(
            raw_params=raw_data,
            run_label=run_label,
            save_to_db=True,
        )
        return jsonify({"status": "success", "data": sim_run})
    except ValidationError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    except (ModelNotTrainedError, FileNotFoundError) as exc:
        return jsonify({"status": "error", "message": str(exc)}), 503


@simulation_bp.route("/scenario-analysis")
def scenario_analysis():
    services = current_app.extensions["services"]
    hours = request.args.get("hours", default=168, type=int)
    hours = max(24, min(720, hours))
    scenario_data = services["analysis"].run_scenario_comparison(hours=hours)
    payload = {
        "active_menu": "scenario_analysis",
        "scenario_data": scenario_data,
    }
    return render_or_fallback("scenario_analysis.html", "Scenario Analysis", payload)


@simulation_bp.route("/sensitivity-analysis")
def sensitivity_analysis():
    services = current_app.extensions["services"]
    param_name = request.args.get("parameter", default="damage_rate", type=str)
    hours = request.args.get("hours", default=168, type=int)
    hours = max(24, min(720, hours))
    sens_data = services["analysis"].run_sensitivity_analysis(parameter_name=param_name, hours=hours)
    payload = {
        "active_menu": "sensitivity_analysis",
        "sensitivity_data": sens_data,
    }
    return render_or_fallback("sensitivity_analysis.html", "Sensitivity Analysis", payload)
