"""Treino, avaliação e salvamento do modelo."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import Pipeline

from triage.config import DATA_PATH, MODEL_DIR, MODEL_VERSION, SEED, URGENT_THRESHOLD
from triage.data import clean_dataset, load_dataset, split_dataset
from triage.labels import LEVELS
from triage.model import build_pipeline
from triage.predict import predict_labels

logger = logging.getLogger(__name__)


def train_model(train_df: pd.DataFrame) -> Pipeline:
    """Treina o pipeline com as frases e os níveis de urgência."""
    pipeline = build_pipeline()
    pipeline.fit(train_df["phrase"], train_df["urgency"])
    return pipeline


def evaluate_model(model: Pipeline, test_df: pd.DataFrame) -> dict:
    """Calcula as métricas do modelo no conjunto de teste."""
    y_true = test_df["urgency"]
    y_pred = predict_labels(model, test_df["phrase"])
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 3),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 3),
        "recall_urgente": round(
            float(recall_score(y_true, y_pred, labels=["urgente"], average="macro")), 3
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LEVELS).tolist(),
    }


def save_model(model: Pipeline, metrics: dict, n_train: int, model_dir: Path = MODEL_DIR) -> Path:
    """Salva o modelo treinado e um arquivo com as informações do treino."""
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)
    metadata = {
        "version": MODEL_VERSION,
        "trained_at": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "urgent_threshold": URGENT_THRESHOLD,
        "n_train": n_train,
        "metrics": metrics,
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = clean_dataset(load_dataset(DATA_PATH))
    train_df, test_df = split_dataset(df)
    model = train_model(train_df)
    logger.info("Modelo treinado com %d frases", len(train_df))
    metrics = evaluate_model(model, test_df)
    logger.info("Métricas: %s", metrics)
    model_path = save_model(model, metrics, len(train_df))
    logger.info("Modelo salvo em %s", model_path)
