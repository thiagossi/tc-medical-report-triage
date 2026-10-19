"""Testes da regra de decisão e da predição de uma frase."""

import numpy as np

from triage.config import URGENT_THRESHOLD
from triage.labels import LEVELS
from triage.predict import apply_threshold, predict_text

CLASSES = np.array(["atencao", "normal", "urgente"])  # ordem alfabética, como no sklearn


def test_most_likely_level_wins_when_urgent_is_low():
    probas = np.array([[0.7, 0.2, 0.1]])
    assert apply_threshold(probas, CLASSES)[0] == "atencao"


def test_urgent_wins_when_above_threshold_even_if_not_the_highest():
    low = 1 - URGENT_THRESHOLD
    probas = np.array([[0.0, low, URGENT_THRESHOLD]])
    assert apply_threshold(probas, CLASSES)[0] == "urgente"


def test_predict_text_returns_level_and_probabilities(trained_model):
    result = predict_text(trained_model, "My heart hurts a lot")
    assert result["label"] in LEVELS
    assert set(result["probabilities"]) == set(LEVELS)
    assert abs(sum(result["probabilities"].values()) - 1) < 0.01
    assert result["confidence"] == result["probabilities"][result["label"]]
