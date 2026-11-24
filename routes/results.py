from flask import Blueprint, Response, current_app, request
from routes import render_or_fallback

results_bp = Blueprint("results", __name__)


@results_bp.route("/results")
def results_and_reports():
    services = current_app.extensions["services"]
    selected_run_id = request.args.get("run_id", type=int)

    if selected_run_id is not None:
        active_run = services["simulation"].get_run_by_id(selected_run_id)
    else:
        active_run = None

    if active_run is None:
        active_run = services["simulation"].get_latest_or_baseline_run()

    history_runs = services["simulation"].list_recent_runs(limit=15)
    scenario_data = services["analysis"].run_scenario_comparison(
        hours=active_run["parameters"].get("simulation_hours", 168)
    )

    payload = {
        "active_menu": "results",
        "active_run": active_run,
        "history_runs": history_runs,
        "scenario_data": scenario_data,
    }
    return render_or_fallback("results.html", "Results and Reports", payload)


@results_bp.route("/results/export-csv")
def export_results_csv():
    services = current_app.extensions["services"]
    run_id = request.args.get("run_id", type=int)
    csv_content, filename = services["analysis"].export_simulation_csv(run_id=run_id)
    return Response(
        csv_content,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
