"""Regra de decisão sobre as probabilidades do modelo."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from triage.config import URGENT_THRESHOLD


def predict_labels(model: Pipeline, texts: pd.Series | list[str]) -> np.ndarray:
    """Prevê o nível, priorizando urgente quando a chance passa do limiar."""
    probas = model.predict_proba(texts)
    labels = model.classes_[probas.argmax(axis=1)]
    urgent_col = list(model.classes_).index("urgente")
    return np.where(probas[:, urgent_col] >= URGENT_THRESHOLD, "urgente", labels)
