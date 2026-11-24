from flask import Flask, jsonify
from config import Config
from models.demand_predictor import DemandPredictor
from models.scenarios import ScenarioRunner
from models.system_dynamics import SystemDynamicsSimulator
from routes.analysis import analysis_bp
from routes.dashboard import dashboard_bp
from routes.dataset import dataset_bp
from routes.results import results_bp
from routes.simulation import simulation_bp
from services.analysis_service import AnalysisService
from services.data_service import DataService
from services.prediction_service import PredictionService
from services.simulation_service import SimulationService


def create_app(config_object=Config) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(config_object.TEMPLATES_DIR),
        static_folder=str(config_object.STATIC_DIR),
    )
    app.config.from_object(config_object)

    # Pastikan folder instance tersedia untuk SQLite
    config_object.INSTANCE_DIR.mkdir(parents=True, exist_ok=True)

    # Inisialisasi lapisan Models (memuat artefak terlatih dari notebook, tanpa training ulang)
    predictor = DemandPredictor(
        model_path=config_object.MODEL_PATH,
        pipeline_path=config_object.PIPELINE_PATH,
        feature_config_path=config_object.FEATURE_CONFIG_PATH,
    )
    simulator = SystemDynamicsSimulator()
    scenario_runner = ScenarioRunner(simulator=simulator)

    # Inisialisasi lapisan Services
    data_service = DataService(
        raw_path=config_object.RAW_DATA_PATH,
        processed_path=config_object.PROCESSED_DATA_PATH,
        eda_summary_path=config_object.EDA_SUMMARY_PATH,
        prep_summary_path=config_object.PREP_SUMMARY_PATH,
    )
    prediction_service = PredictionService(
        predictor=predictor,
        metrics_path=config_object.METRICS_PATH,
        sim_horizon_path=config_object.SIM_HORIZON_PATH,
        processed_data_path=config_object.PROCESSED_DATA_PATH,
    )
    simulation_service = SimulationService(
        simulator=simulator,
        prediction_service=prediction_service,
        assumptions_path=config_object.SD_ASSUMPTIONS_PATH,
        db_path=config_object.SQLITE_DB_PATH,
        default_params=config_object.DEFAULT_SD_PARAMS,
    )
    analysis_service = AnalysisService(
        simulator=simulator,
        scenario_runner=scenario_runner,
        prediction_service=prediction_service,
        simulation_service=simulation_service,
        scenario_json_path=config_object.SCENARIO_JSON_PATH,
    )

    app.extensions["services"] = {
        "predictor": predictor,
        "simulator": simulator,
        "data": data_service,
        "prediction": prediction_service,
        "simulation": simulation_service,
        "analysis": analysis_service,
    }

    # Registrasi Blueprints
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(dataset_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(simulation_bp)
    app.register_blueprint(results_bp)

    @app.context_processor
    def inject_global_context():
        return {
            "app_name": config_object.APP_NAME,
            "app_tagline": config_object.APP_TAGLINE,
            "research_title": config_object.RESEARCH_TITLE,
            "global_model_status": predictor.get_status(),
        }

    @app.route("/health")
    def health_check():
        return jsonify(
            {
                "status": "ok",
                "app": config_object.APP_NAME,
                "model_status": predictor.get_status(),
            }
        )

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
