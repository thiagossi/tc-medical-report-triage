"""API de triagem: recebe uma frase e devolve o nível de urgência."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response, status

from triage.api.metrics import (
    MODEL_INFO,
    PREDICTIONS,
    metrics_endpoint,
    metrics_middleware,
)
from triage.api.schemas import PredictRequest, PredictResponse
from triage.config import MODEL_DIR
from triage.predict import load_model, predict_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

# Guarda o modelo carregado enquanto a API estiver ligada.
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Carrega o modelo uma única vez, quando a API liga."""
    try:
        state["model"], state["metadata"] = load_model()
        logger.info(
            "Modelo %s (%s) carregado de %s",
            state["metadata"]["version"],
            state["metadata"].get("backend"),
            MODEL_DIR,
        )
        MODEL_INFO.labels(state["metadata"]["version"]).set(1)
    except FileNotFoundError:
        logger.error("Modelo não encontrado em %s. Rode o treino antes.", MODEL_DIR)
    yield
    state.clear()


app = FastAPI(title="Triagem de relatos médicos", version="1.0.0", lifespan=lifespan)
app.middleware("http")(metrics_middleware)
app.add_api_route("/metrics", metrics_endpoint, methods=["GET"], include_in_schema=False)


@app.get("/health")
def health(response: Response) -> dict:
    """Diz se a API está no ar e se o modelo foi carregado."""
    loaded = "model" in state
    if not loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ok" if loaded else "modelo indisponível",
        "model_version": state["metadata"]["version"] if loaded else None,
        "backend": state["metadata"].get("backend") if loaded else None,
    }


@app.post("/predict")
def predict(request: PredictRequest) -> PredictResponse:
    """Classifica a frase em normal, atenção ou urgente."""
    if "model" not in state:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Modelo não carregado")
    result = predict_text(state["model"], request.text)
    PREDICTIONS.labels(result["label"]).inc()
    return PredictResponse(**result, model_version=state["metadata"]["version"])
