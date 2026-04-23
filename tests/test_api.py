from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.utils.config import settings


def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {settings.api_token}"}


def test_health_and_readiness_are_public() -> None:
    client = TestClient(app)

    health_response = client.get("/health")
    readiness_response = client.get("/readiness")

    assert health_response.status_code == 200
    assert readiness_response.status_code == 200
    assert health_response.json()["backend_auth_enabled"] is True
    assert readiness_response.json()["llm_runtime_mode"] == "heuristic"


def test_profile_requires_api_token() -> None:
    client = TestClient(app)

    unauthorized = client.get("/profile")
    authorized = client.get("/profile", headers=auth_headers())

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200


def test_mock_send_returns_conflict_until_run_is_approved() -> None:
    client = TestClient(app)
    run = client.post(
        "/analyze",
        json={"raw_message": "Hi Alex, we would love to chat about a backend role."},
        headers=auth_headers(),
    ).json()

    response = client.post(
        f"/runs/{run['run_id']}/mock-send",
        json={"edited_draft": "Sending too early."},
        headers=auth_headers(),
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "mock_send_requires_approval"
