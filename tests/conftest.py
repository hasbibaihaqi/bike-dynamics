import pytest
from app import create_app
from config import Config


@pytest.fixture()
def app(tmp_path):
    class TestConfig(Config):
        TESTING = True
        SQLITE_DB_PATH = tmp_path / "test_bike_dynamics.db"

    flask_app = create_app(TestConfig)
    yield flask_app


@pytest.fixture()
def client(app):
    return app.test_client()
