from flask import Blueprint, current_app
from routes import render_or_fallback

dataset_bp = Blueprint("dataset", __name__)


@dataset_bp.route("/dataset")
def dataset_management():
    services = current_app.extensions["services"]
    dataset_info = services["data"].get_dataset_overview(preview_rows=15)
    payload = {
        "active_menu": "dataset",
        "dataset_info": dataset_info,
    }
    return render_or_fallback("dataset.html", "Dataset Management", payload)
