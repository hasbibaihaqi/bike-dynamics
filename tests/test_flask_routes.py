import pytest


@pytest.mark.parametrize(
    "route_path,expected_text",
    [
        ("/", "Total Armada"),
        ("/dataset", "Dataset Management"),
        ("/demand-analysis", "Demand Analysis"),
        ("/model-performance", "Model Performance"),
        ("/system-dynamics", "System Dynamics"),
        ("/simulation-center", "Simulation Center"),
        ("/scenario-analysis", "Scenario Analysis"),
        ("/sensitivity-analysis", "Sensitivity Analysis"),
        ("/results", "Results and Reports"),
    ],
)
def test_all_nine_menus_return_200(client, route_path, expected_text):
    resp = client.get(route_path)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert expected_text in html


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["status"] == "ok"
    assert payload["model_status"]["ready"] is True


def test_api_predict_demand(client):
    resp = client.post(
        "/api/predict-demand",
        json={
            "season": 3,
            "weather": 1,
            "year": 2012,
            "month": 9,
            "day": 10,
            "hour": 17,
            "dayofweek": 1,
            "holiday": 0,
            "workingday": 1,
            "temp": 26.0,
            "atemp": 28.0,
            "humidity": 50.0,
            "windspeed": 10.0,
        },
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "success"
    assert body["data"]["predicted_demand"] >= 0.0

    bad_resp = client.post("/api/predict-demand", json={"hour": 99})
    assert bad_resp.status_code == 400


def test_simulation_center_post_and_csv_export(client):
    post_resp = client.post(
        "/simulation-center",
        data={
            "run_label": "Uji Pytest Armada 650",
            "total_fleet": "650",
            "initial_available": "550",
            "initial_maintenance": "30",
            "rental_duration_hours": "1.5",
            "damage_rate": "0.04",
            "repair_duration_hours": "5.0",
            "demand_multiplier": "1.0",
            "simulation_hours": "48",
        },
    )
    assert post_resp.status_code == 200
    assert "Uji Pytest Armada 650" in post_resp.get_data(as_text=True)

    csv_resp = client.get("/results/export-csv")
    assert csv_resp.status_code == 200
    assert "text/csv" in csv_resp.content_type
    csv_text = csv_resp.get_data(as_text=True)
    assert "step,datetime,weather,demand_rate,rental_rate,unmet_demand" in csv_text
