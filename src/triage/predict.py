"""Carregamento do modelo e regra de decisão sobre as probabilidades."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from triage.config import MODEL_DIR, URGENT_THRESHOLD


def load_model(model_dir: Path = MODEL_DIR) -> tuple[Pipeline, dict]:
    """Carrega do disco o modelo treinado e os seus metadados."""
    model = joblib.load(model_dir / "model.joblib")
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    return model, metadata


def apply_threshold(probas: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Escolhe o nível mais provável, mas prioriza urgente quando a chance passa do limiar."""
    labels = classes[probas.argmax(axis=1)]
    urgent_col = list(classes).index("urgente")
    return np.where(probas[:, urgent_col] >= URGENT_THRESHOLD, "urgente", labels)


def predict_labels(model: Pipeline, texts: pd.Series | list[str]) -> np.ndarray:
    """Prevê o nível de várias frases (usado na avaliação do treino)."""
    return apply_threshold(model.predict_proba(texts), model.classes_)


def predict_text(model: Pipeline, text: str) -> dict:
    """Prevê o nível de uma frase e devolve a chance de cada nível (usado pela API)."""
    probas = model.predict_proba([text])
    label = str(apply_threshold(probas, model.classes_)[0])
    probabilities = {
        str(level): round(float(p), 4) for level, p in zip(model.classes_, probas[0], strict=True)
    }
    return {"label": label, "confidence": probabilities[label], "probabilities": probabilities}
