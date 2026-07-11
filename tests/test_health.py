from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get(f"{settings.API_V1_STR}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["project"] == settings.PROJECT_NAME
