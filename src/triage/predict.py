"""Carregamento do modelo e regra de decisão sobre as probabilidades."""

import json
import re
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.pipeline import Pipeline

from triage.config import MODEL_BACKEND, MODEL_DIR, URGENT_THRESHOLD

# O TF-IDF de letras do sklearn troca espaços repetidos por um só antes de quebrar o texto.
# O ONNX não faz isso, então fazemos aqui para as duas versões verem o mesmo texto.
MULTIPLE_SPACES = re.compile(r"\s\s+")


class OnnxModel:
    """Modelo ONNX com a mesma "cara" do sklearn: predict_proba e classes_.

    Assim o resto do código (regra de decisão, API) funciona igual com os dois.
    """

    def __init__(self, onnx_model: bytes | Path, classes: list[str]):
        source = str(onnx_model) if isinstance(onnx_model, Path) else onnx_model
        # Uma frase por vez é pouco trabalho: dividir entre várias threads só cria disputa
        # pela CPU (no Docker com 1 CPU, o p95 ia de 3 ms para 76 ms). Por isso, 1 thread.
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            source, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.classes_ = np.array(classes)

    def predict_proba(self, texts: pd.Series | list[str]) -> np.ndarray:
        # O ONNX espera uma coluna de textos: formato (n, 1).
        cleaned = [MULTIPLE_SPACES.sub(" ", text) for text in texts]
        inputs = np.asarray(cleaned, dtype=object).reshape(-1, 1)
        _, probas = self.session.run(None, {"text": inputs})
        # O ONNX calcula em float32: 0.25 pode virar 0.2499999 e escapar do limiar de urgente.
        # Arredondar em 6 casas devolve o mesmo valor do sklearn (float64).
        return np.round(probas.astype(np.float64), 6)


def load_model(
    model_dir: Path = MODEL_DIR, backend: str = MODEL_BACKEND
) -> tuple[Pipeline | OnnxModel, dict]:
    """Carrega do disco o modelo treinado (sklearn ou ONNX) e os seus metadados."""
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    if backend == "sklearn":
        model = joblib.load(model_dir / "model.joblib")
    elif backend == "onnx":
        if "onnx" not in metadata:
            raise FileNotFoundError("Modelo ONNX não exportado. Rode: python -m triage.optimize")
        onnx_info = metadata["onnx"]
        model = OnnxModel(model_dir / onnx_info["file"], onnx_info["classes"])
    else:
        raise ValueError(f'MODEL_BACKEND deve ser "sklearn" ou "onnx", não "{backend}"')
    metadata["backend"] = backend
    return model, metadata


def apply_threshold(probas: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Escolhe o nível mais provável, mas prioriza urgente quando a chance passa do limiar."""
    labels = classes[probas.argmax(axis=1)]
    urgent_col = list(classes).index("urgente")
    return np.where(probas[:, urgent_col] >= URGENT_THRESHOLD, "urgente", labels)


def predict_labels(model: Pipeline | OnnxModel, texts: pd.Series | list[str]) -> np.ndarray:
    """Prevê o nível de várias frases (usado na avaliação do treino)."""
    return apply_threshold(model.predict_proba(texts), model.classes_)


def predict_text(model: Pipeline | OnnxModel, text: str) -> dict:
    """Prevê o nível de uma frase e devolve a chance de cada nível (usado pela API)."""
    probas = model.predict_proba([text])
    label = str(apply_threshold(probas, model.classes_)[0])
    probabilities = {
        str(level): round(float(p), 4) for level, p in zip(model.classes_, probas[0], strict=True)
    }
    return {"label": label, "confidence": probabilities[label], "probabilities": probabilities}
