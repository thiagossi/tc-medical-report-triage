"""Treino, avaliação e salvamento do modelo."""

import logging

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import Pipeline

from triage.config import DATA_PATH
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = clean_dataset(load_dataset(DATA_PATH))
    train_df, test_df = split_dataset(df)
    model = train_model(train_df)
    logger.info("Modelo treinado com %d frases", len(train_df))
    metrics = evaluate_model(model, test_df)
    logger.info("Métricas: %s", metrics)
