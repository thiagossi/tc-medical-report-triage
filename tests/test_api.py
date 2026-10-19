"""Testes dos endpoints da API."""

import pytest
from fastapi.testclient import TestClient

from triage.api import main


@pytest.fixture
def client(monkeypatch, trained_model):
    """API ligada com o modelo de teste (sem ler nada do disco)."""
    monkeypatch.setattr(main, "load_model", lambda: (trained_model, {"version": "test"}))
    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture
def client_without_model(monkeypatch):
    """API ligada sem modelo, como se o treino ainda não tivesse rodado."""

    def missing_model():
        raise FileNotFoundError

    monkeypatch.setattr(main, "load_model", missing_model)
    with TestClient(main.app) as test_client:
        yield test_client


def test_health_ok_when_model_is_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_version"] == "test"


def test_predict_returns_a_level(client):
    response = client.post("/predict", json={"text": "My heart hurts"})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in {"normal", "atencao", "urgente"}
    assert body["model_version"] == "test"


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": "   "}, {"text": "a" * 1001}, {}])
def test_predict_rejects_invalid_input(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_without_model_api_answers_503(client_without_model):
    assert client_without_model.get("/health").status_code == 503
    response = client_without_model.post("/predict", json={"text": "My heart hurts"})
    assert response.status_code == 503
