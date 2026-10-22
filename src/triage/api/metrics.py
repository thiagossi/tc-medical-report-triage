"""Métricas Prometheus da API, expostas em /metrics."""

import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from triage.labels import LEVELS

REQUESTS = Counter(
    "http_requests_total",
    "Total de requisições HTTP recebidas",
    ["method", "endpoint", "status"],
)
LATENCY = Histogram(
    "http_request_duration_seconds",
    "Tempo de resposta das requisições HTTP, em segundos",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0),
)
PREDICTIONS = Counter(
    "predictions_total",
    "Predições feitas, por nível de urgência",
    ["label"],
)
MODEL_INFO = Gauge(
    "model_info",
    "Versão do modelo carregado (vale 1 para a versão em uso)",
    ["version"],
)

# Cria as séries dos 3 níveis com zero, para o painel aparecer antes da 1ª predição.
for level in LEVELS:
    PREDICTIONS.labels(level)


def endpoint_of(request: Request) -> str:
    """Rota "modelo" (ex.: /predict), e não a URL exata, para não explodir o nº de séries."""
    route = request.scope.get("route")
    return getattr(route, "path", "nao_encontrada")


async def metrics_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Mede o tempo e conta cada requisição, inclusive as que dão erro."""
    if request.url.path == "/metrics":
        return await call_next(request)

    start = time.perf_counter()
    status_code = 500  # se der uma exceção não tratada, conta como erro 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        endpoint = endpoint_of(request)
        LATENCY.labels(request.method, endpoint).observe(time.perf_counter() - start)
        REQUESTS.labels(request.method, endpoint, str(status_code)).inc()


def metrics_endpoint() -> Response:
    """Texto no formato que o Prometheus lê."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
