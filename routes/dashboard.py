from flask import Blueprint, current_app
from routes import render_or_fallback

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/")
def index():
    services = current_app.extensions["services"]
    latest_run = services["simulation"].get_latest_or_baseline_run()
    summary = latest_run["summary"]
    timeseries = latest_run["timeseries"]

    # Ambil kondisi snapshot jam terakhir & ringkasan agregat
    last_point = timeseries[-1] if timeseries else {}
    kpi = {
        "total_fleet": summary.get("total_fleet", 0),
        "available_bikes": last_point.get("available_bikes", summary.get("avg_available_bikes", 0)),
        "rented_bikes": last_point.get("rented_bikes", summary.get("avg_rented_bikes", 0)),
        "maintenance_bikes": last_point.get("maintenance_bikes", summary.get("avg_maintenance_bikes", 0)),
        "avg_available_bikes": summary.get("avg_available_bikes", 0),
        "avg_rented_bikes": summary.get("avg_rented_bikes", 0),
        "avg_maintenance_bikes": summary.get("avg_maintenance_bikes", 0),
        "total_demand": summary.get("total_demand", 0),
        "total_rentals": summary.get("total_rentals", 0),
        "total_unmet_demand": summary.get("total_unmet_demand", 0),
        "avg_utilization_pct": summary.get("avg_utilization_pct", 0),
        "service_level_pct": summary.get("service_level_pct", 0),
    }

    payload = {
        "active_menu": "dashboard",
        "kpi": kpi,
        "latest_run": latest_run,
        "model_status": services["predictor"].get_status(),
    }
    return render_or_fallback("dashboard.html", "Dashboard", payload)
