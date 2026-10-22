"""Testes das métricas Prometheus expostas pela API."""

import pytest
from fastapi.testclient import TestClient

from triage.api import main


@pytest.fixture
def client(monkeypatch, trained_model):
    monkeypatch.setattr(main, "load_model", lambda: (trained_model, {"version": "test"}))
    with TestClient(main.app) as test_client:
        yield test_client


def test_metrics_endpoint_is_exposed(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text


def test_requests_errors_and_predictions_are_counted(client):
    client.post("/predict", json={"text": "My heart hurts"})
    client.post("/predict", json={"text": ""})  # erro 422

    text = client.get("/metrics").text

    assert 'http_requests_total{endpoint="/predict",method="POST",status="200"}' in text
    assert 'http_requests_total{endpoint="/predict",method="POST",status="422"}' in text
    assert "http_request_duration_seconds_bucket" in text
    assert 'predictions_total{label="' in text
    assert 'model_info{version="test"} 1.0' in text


def test_metrics_endpoint_does_not_count_itself(client):
    client.get("/metrics")
    assert 'endpoint="/metrics"' not in client.get("/metrics").text
